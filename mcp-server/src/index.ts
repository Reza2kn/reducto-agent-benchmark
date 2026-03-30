#!/usr/bin/env node

/**
 * Reducto MCP Server
 *
 * Exposes Reducto's document intelligence API as MCP tools for any
 * MCP-compatible AI agent (Claude Code, Cursor, Codex, Cline, etc.)
 *
 * Tools:
 *   reducto_parse      — Parse documents into structured markdown, tables, figures
 *   reducto_extract    — Extract specific fields into typed JSON via schema
 *   reducto_split      — Divide documents into named sections by page range
 *   reducto_edit       — Fill forms / modify documents with natural language
 *   reducto_classify   — Classify document type before processing
 *   reducto_upload     — Upload a local file via URL, returns reducto:// file ID
 *   reducto_get_job    — Retrieve results of a previous job (enables jobid:// reuse)
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const BASE_URL = process.env.REDUCTO_BASE_URL ?? "https://platform.reducto.ai";
const API_KEY = process.env.REDUCTO_API_KEY;

if (!API_KEY) {
  console.error(
    "Error: REDUCTO_API_KEY environment variable is required.\n" +
      "Get one at https://studio.reducto.ai/ → API Keys → Create new API key"
  );
  process.exit(1);
}

// ---------------------------------------------------------------------------
// HTTP helper — retries on 429 (rate limit) and 500 (transient)
// ---------------------------------------------------------------------------

async function reductoFetch(
  path: string,
  body: Record<string, unknown>,
  method = "POST"
): Promise<unknown> {
  const maxRetries = 3;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const res = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (res.ok) return res.json();

    // Parse structured errors — API returns { detail: [{ loc, msg, type }] }
    const text = await res.text();
    let message = text;
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed.detail)) {
        message = parsed.detail.map((d: any) => `${d.msg} (at ${d.loc?.join(".")})`).join("; ");
      } else if (parsed.detail) {
        message = String(parsed.detail);
      }
    } catch {
      // keep raw text
    }

    // Retry on rate limit or transient server error
    const retryable = res.status === 429 || res.status === 500;
    if (retryable && attempt < maxRetries) {
      // Respect Retry-After header if present, else exponential backoff
      const retryAfter = res.headers.get("Retry-After");
      const waitMs = retryAfter
        ? parseInt(retryAfter, 10) * 1000
        : Math.pow(2, attempt) * 1000;
      await new Promise((r) => setTimeout(r, waitMs));
      continue;
    }

    throw new Error(`Reducto API error ${res.status}: ${message}`);
  }

  throw new Error("Reducto API: max retries exceeded");
}

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

const server = new McpServer({
  name: "reducto",
  version: "0.2.0",
});

// ---------------------------------------------------------------------------
// Tool: reducto_parse
// ---------------------------------------------------------------------------

const BLOCK_TYPES = [
  "Header", "Footer", "Title", "Section Header", "Page Number",
  "List Item", "Figure", "Table", "Key Value", "Text", "Comment", "Signature",
] as const;

server.tool(
  "reducto_parse",
  "Parse a document (PDF, image, spreadsheet, DOCX, etc.) into structured content " +
    "including text, tables, and figures with bounding boxes. Accepts public URLs, " +
    "presigned S3/GCS/Azure URLs, or reducto:// file IDs from reducto_upload. " +
    "Returns markdown-formatted chunks with block-level type annotations.\n\n" +
    "TIP: Set persist_results=true to get a job ID you can pass as 'jobid://JOB_ID' " +
    "to reducto_extract or reducto_split — avoids re-parsing the same document.",
  {
    input: z
      .string()
      .describe(
        "Document source: a public URL, presigned cloud URL, reducto:// file ID, " +
          "or jobid://JOB_ID to re-use a previous parse result"
      ),
    filter_blocks: z
      .array(z.enum(BLOCK_TYPES))
      .optional()
      .describe(
        "Block types to exclude from output. Common use: ['Header','Footer','Page Number'] " +
          "to remove navigation noise before feeding to an agent or RAG pipeline."
      ),
    merge_tables: z
      .boolean()
      .optional()
      .describe(
        "Merge tables that are split across page boundaries into a single table. " +
          "Default: false. Strongly recommended for financial statements, long reports."
      ),
    extraction_mode: z
      .enum(["ocr", "hybrid"])
      .optional()
      .describe(
        "Text extraction strategy. 'hybrid' (default): native text + OCR fallback. " +
          "'ocr': force full OCR — use for scanned documents or low-quality images."
      ),
    chunk_mode: z
      .enum(["disabled", "variable", "section", "page", "block", "page_sections"])
      .optional()
      .describe(
        "How to chunk output. 'variable' (best for RAG), 'page', 'section', 'block', " +
          "'page_sections', or 'disabled' (single chunk). Default: disabled."
      ),
    table_format: z
      .enum(["dynamic", "html", "md", "json", "csv", "jsonbbox"])
      .optional()
      .describe(
        "Table output format. 'dynamic' auto-selects. 'html' best for complex merged-cell tables. " +
          "'jsonbbox' includes bounding boxes per cell."
      ),
    add_page_markers: z
      .boolean()
      .optional()
      .describe(
        "Insert '## Page N' markers between pages in the output. " +
          "Useful when you need to know which page each piece of content came from."
      ),
    agentic_scopes: z
      .array(z.enum(["text", "table", "figure"]))
      .optional()
      .describe(
        "Enable AI-powered correction. 'text': OCR cleanup, 'table': fix misaligned columns, " +
          "'figure': extract chart data as structured JSON. Adds latency/cost."
      ),
    return_figure_images: z
      .boolean()
      .optional()
      .describe(
        "Return presigned image URLs for every figure detected in the document. " +
          "Useful when you need to see charts or diagrams, not just their extracted text."
      ),
    persist_results: z
      .boolean()
      .optional()
      .describe(
        "Keep this parse result accessible for 1 hour so other tools can reference it " +
          "as 'jobid://JOB_ID' without re-parsing. Recommended for multi-step workflows."
      ),
    page_range_start: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("First page to process (1-indexed). Use with page_range_end to target a section."),
    page_range_end: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("Last page to process (1-indexed)."),
  },
  async (params) => {
    const body: Record<string, unknown> = { input: params.input };

    // Agentic enhancement
    if (params.agentic_scopes?.length) {
      body.enhance = {
        agentic: params.agentic_scopes.map((scope) => ({ scope })),
      };
    }

    // Retrieval / chunking / filtering
    const retrieval: Record<string, unknown> = {};
    if (params.chunk_mode) {
      retrieval.chunking = { chunk_mode: params.chunk_mode };
    }
    if (params.filter_blocks?.length) {
      retrieval.filter_blocks = params.filter_blocks;
    }
    if (Object.keys(retrieval).length) body.retrieval = retrieval;

    // Formatting
    const formatting: Record<string, unknown> = {};
    if (params.table_format) formatting.table_output_format = params.table_format;
    if (params.merge_tables != null) formatting.merge_tables = params.merge_tables;
    if (params.add_page_markers != null) formatting.add_page_markers = params.add_page_markers;
    if (Object.keys(formatting).length) body.formatting = formatting;

    // Settings
    const settings: Record<string, unknown> = {};
    if (params.page_range_start || params.page_range_end) {
      settings.page_range = {
        ...(params.page_range_start && { start: params.page_range_start }),
        ...(params.page_range_end && { end: params.page_range_end }),
      };
    }
    if (params.extraction_mode) settings.extraction_mode = params.extraction_mode;
    if (params.return_figure_images) settings.return_images = ["figure"];
    if (params.persist_results != null) settings.persist_results = params.persist_results;
    if (Object.keys(settings).length) body.settings = settings;

    const result = (await reductoFetch("/parse", body)) as any;

    // Large docs: API returns a URL instead of inline chunks
    let chunks = result.result?.chunks;
    if (result.result?.type === "url" && result.result?.url) {
      const urlRes = await fetch(result.result.url);
      chunks = await urlRes.json();
    }

    // Summary stats for agent situational awareness
    const tableCount = chunks?.reduce(
      (n: number, c: any) =>
        n + (c.blocks?.filter((b: any) => b.type === "Table").length ?? 0),
      0
    ) ?? 0;
    const figureCount = chunks?.reduce(
      (n: number, c: any) =>
        n + (c.blocks?.filter((b: any) => b.type === "Figure").length ?? 0),
      0
    ) ?? 0;

    const header =
      `Parsed ${result.usage?.num_pages ?? "?"} pages ` +
      `| ${tableCount} tables, ${figureCount} figures ` +
      `| ${result.usage?.credits ?? "?"} credits, ${result.duration?.toFixed(1) ?? "?"}s\n` +
      `Job ID: ${result.job_id}` +
      (params.persist_results
        ? `\n💡 Re-use this parse: pass input='jobid://${result.job_id}' to reducto_extract or reducto_split`
        : "");

    const content =
      chunks
        ?.map((c: any, i: number) => {
          const text = typeof c === "string" ? c : c.content ?? JSON.stringify(c);
          return `--- Chunk ${i + 1} ---\n${text}`;
        })
        .join("\n\n") ?? "No chunks returned.";

    return {
      content: [{ type: "text" as const, text: `${header}\n\n${content}` }],
    };
  }
);

// ---------------------------------------------------------------------------
// Tool: reducto_extract
// ---------------------------------------------------------------------------

server.tool(
  "reducto_extract",
  "Extract specific fields from a document into structured JSON using a provided " +
    "JSON Schema. Great for invoices, forms, contracts, financial statements — any " +
    "document where you need specific typed fields. Returns an array of extracted objects.\n\n" +
    "TIP: Pass a previous parse job as 'jobid://JOB_ID' to skip re-parsing. " +
    "Use deep_extract=true for difficult documents where initial extraction misses fields. " +
    "Enable citations=true to get bounding boxes showing exactly where each value was found.",
  {
    input: z
      .string()
      .describe(
        "Document source: URL, reducto:// file ID, or jobid://JOB_ID to reuse a previous parse"
      ),
    schema: z
      .string()
      .describe(
        "JSON Schema string defining the fields to extract. " +
          "Add 'description' to each field — the model uses these as extraction hints. " +
          'Example: \'{"type":"object","properties":{"invoice_number":{"type":"string","description":"Invoice ID at top of document"},"total":{"type":"number","description":"Final amount due including tax"}}}\''
      ),
    system_prompt: z
      .string()
      .optional()
      .describe(
        "Custom instructions for the extraction model. Use this to handle edge cases, " +
          "specify units/formats, or guide ambiguous fields. " +
          "Example: 'All monetary values are in USD. Use null if a field is not present.'"
      ),
    array_extract: z
      .boolean()
      .optional()
      .describe(
        "Extract a repeated structure (e.g. line items from an invoice). " +
          "When true, schema must have at least one top-level array property. Default: false."
      ),
    deep_extract: z
      .boolean()
      .optional()
      .describe(
        "Run an agentic refinement pass after initial extraction to improve accuracy " +
          "on complex or ambiguous documents. Higher cost and latency. " +
          "Use when initial extraction misses or mis-formats fields."
      ),
    citations: z
      .boolean()
      .optional()
      .describe(
        "Return bounding box citations for each extracted value — shows exactly where " +
          "on the page each field was found, with confidence scores. " +
          "Useful for verification workflows. Cannot be combined with chunking."
      ),
    include_images: z
      .boolean()
      .optional()
      .describe(
        "Include figure and chart images in the extraction context. " +
          "Enables extracting data from charts, diagrams, or image-heavy documents."
      ),
  },
  async (params) => {
    let schema: unknown;
    try {
      schema = JSON.parse(params.schema);
    } catch {
      return {
        content: [{ type: "text" as const, text: "Error: 'schema' must be a valid JSON string." }],
        isError: true,
      };
    }

    const instructions: Record<string, unknown> = { schema };
    if (params.system_prompt) instructions.system_prompt = params.system_prompt;

    const settings: Record<string, unknown> = {};
    if (params.array_extract) settings.array_extract = true;
    if (params.deep_extract) settings.deep_extract = true;
    if (params.citations) settings.citations = { enabled: true, numerical_confidence: true };
    if (params.include_images) settings.include_images = true;

    const body: Record<string, unknown> = { input: params.input, instructions };
    if (Object.keys(settings).length) body.settings = settings;

    const result = (await reductoFetch("/extract", body)) as any;

    return {
      content: [
        {
          type: "text" as const,
          text:
            `Extraction complete | ${result.usage?.num_pages ?? "?"} pages, ` +
            `${result.usage?.credits ?? "?"} credits\n` +
            `Job ID: ${result.job_id}\n\n` +
            JSON.stringify(result.result, null, 2),
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// Tool: reducto_split
// ---------------------------------------------------------------------------

server.tool(
  "reducto_split",
  "Divide a document into named logical sections by page range. Useful for separating " +
    "a long document into individually processable parts (e.g. 'Cover Letter', " +
    "'Financial Statements', 'Appendix'). Returns page ranges per section.\n\n" +
    "TIP: Pass the returned page ranges back to reducto_parse with page_range_start/end " +
    "to parse only the relevant section of a large document.",
  {
    input: z
      .string()
      .describe("Document source: URL, reducto:// file ID, or jobid://JOB_ID"),
    split_description: z
      .string()
      .describe(
        "JSON array of section definitions. Each item needs 'name' and 'description'. " +
          "Example: '[{\"name\":\"Executive Summary\",\"description\":\"High-level overview, typically 1-2 pages\"},{\"name\":\"Financials\",\"description\":\"Balance sheet, P&L, cash flow tables\"}]'"
      ),
    split_rules: z
      .string()
      .optional()
      .describe(
        "Natural language override rules for how to split. Use when the default splitting " +
          "makes wrong cuts. Example: 'Never split a table across sections. " +
          "If a section is missing, mark it as empty rather than merging with adjacent.'"
      ),
    table_cutoff: z
      .enum(["truncate", "preserve"])
      .optional()
      .describe(
        "What to do when a section boundary falls inside a table. " +
          "'truncate' (default): cut the table at the boundary. " +
          "'preserve': extend the section to include the full table."
      ),
  },
  async (params) => {
    let splitDesc: unknown;
    try {
      splitDesc = JSON.parse(params.split_description);
    } catch {
      return {
        content: [{ type: "text" as const, text: "Error: 'split_description' must be a valid JSON array string." }],
        isError: true,
      };
    }

    const body: Record<string, unknown> = {
      input: params.input,
      split_description: splitDesc,
    };
    if (params.split_rules) body.split_rules = params.split_rules;
    if (params.table_cutoff) body.settings = { table_cutoff: params.table_cutoff };

    const result = (await reductoFetch("/split", body)) as any;

    const splits = result.result?.splits
      ?.map(
        (s: any) =>
          `• ${s.name}: pages ${JSON.stringify(s.pages)} (confidence: ${s.conf ?? "N/A"})`
      )
      .join("\n");

    return {
      content: [
        {
          type: "text" as const,
          text:
            `Split complete | ${result.usage?.num_pages ?? "?"} pages, ` +
            `${result.usage?.credits ?? "?"} credits, ${result.duration?.toFixed(1) ?? "?"}s\n` +
            `Job ID: ${result.job_id}\n\n` +
            (splits ?? "No splits returned.") +
            "\n\n💡 Use page ranges above with reducto_parse(page_range_start, page_range_end) to process each section individually.",
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// Tool: reducto_edit
// ---------------------------------------------------------------------------

server.tool(
  "reducto_edit",
  "Fill forms or modify a document using natural language instructions. " +
    "Returns a presigned URL to the edited document (valid 24 hours). Works with PDFs and DOCX.",
  {
    document_url: z
      .string()
      .describe("Document source: URL or reducto:// file ID"),
    edit_instructions: z
      .string()
      .describe(
        "Natural language instructions for editing. " +
          "Example: \"Fill Name: John Doe, Date: 2024-01-15, check the 'US Citizen' box\""
      ),
    highlight_color: z
      .string()
      .optional()
      .describe(
        "Hex color for filled fields, e.g. '#FFFF00' for yellow highlight. Default: red (#FF0000)."
      ),
    enable_overflow_pages: z
      .boolean()
      .optional()
      .describe(
        "Allow adding new pages if the filled content overflows the original page. Default: false."
      ),
  },
  async (params) => {
    const body: Record<string, unknown> = {
      document_url: params.document_url,
      edit_instructions: params.edit_instructions,
    };

    const editOptions: Record<string, unknown> = {};
    if (params.highlight_color) editOptions.color = params.highlight_color;
    if (params.enable_overflow_pages != null) editOptions.enable_overflow_pages = params.enable_overflow_pages;
    if (Object.keys(editOptions).length) body.edit_options = editOptions;

    const result = (await reductoFetch("/edit", body)) as any;

    return {
      content: [
        {
          type: "text" as const,
          text:
            `Edit complete | ${result.usage?.num_pages ?? "?"} pages, ` +
            `${result.usage?.credits ?? "?"} credits\n` +
            `Job ID: ${result.job_id}\n` +
            `Edited document URL (valid 24h): ${result.document_url ?? "N/A"}`,
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// Tool: reducto_classify
// ---------------------------------------------------------------------------

server.tool(
  "reducto_classify",
  "Classify a document's type against a provided set of categories. " +
    "Use this before processing to route documents to the right pipeline " +
    "(e.g. invoice → extract with invoice schema, contract → extract with contract schema).",
  {
    input: z
      .string()
      .describe("Document source: URL or reducto:// file ID"),
    classification_schema: z
      .string()
      .describe(
        "JSON array of categories with criteria. " +
          'Example: \'[{"category":"invoice","criteria":["billing info","itemized charges","due date"]},{"category":"contract","criteria":["parties","obligations","signatures"]}]\''
      ),
    document_metadata: z
      .string()
      .optional()
      .describe(
        "Optional context string about the document to help the classifier. " +
          "Example: 'This document came from an AP automation pipeline and is expected to be either an invoice or a PO.'"
      ),
    page_range_start: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("First page to classify (1-indexed). Default: page 1. Max range: 10 pages."),
    page_range_end: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("Last page to classify (1-indexed). Default: page 5."),
  },
  async (params) => {
    let classSchema: unknown;
    try {
      classSchema = JSON.parse(params.classification_schema);
    } catch {
      return {
        content: [{ type: "text" as const, text: "Error: 'classification_schema' must be a valid JSON array string." }],
        isError: true,
      };
    }

    const body: Record<string, unknown> = {
      input: params.input,
      classification_schema: classSchema,
    };
    if (params.document_metadata) body.document_metadata = params.document_metadata;
    if (params.page_range_start || params.page_range_end) {
      body.page_range = {
        ...(params.page_range_start && { start: params.page_range_start }),
        ...(params.page_range_end && { end: params.page_range_end }),
      };
    }

    const result = (await reductoFetch("/classify", body)) as any;

    return {
      content: [
        {
          type: "text" as const,
          text:
            `Classification: ${result.result?.category ?? "unknown"}\n` +
            `Duration: ${result.duration?.toFixed(1) ?? "?"}s | Job ID: ${result.job_id}\n\n` +
            JSON.stringify(result.result, null, 2),
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// Tool: reducto_upload
// ---------------------------------------------------------------------------

server.tool(
  "reducto_upload",
  "Upload a document to Reducto by URL and get a reducto:// file ID for use with other tools. " +
    "Useful when the original URL has a short-lived signature or you want a stable reference. " +
    "The returned file ID can be passed to any other reducto_* tool as the 'input' parameter.",
  {
    file_url: z
      .string()
      .describe(
        "Public or presigned URL of the file to upload. " +
          "Supported: PDF, DOCX, XLSX, PPTX, PNG, JPG, TIFF, and 20+ other formats. " +
          "Max size: 100MB via URL upload."
      ),
  },
  async (params) => {
    const res = await fetch(`${BASE_URL}/upload`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url: params.file_url }),
    });

    if (!res.ok) {
      const text = await res.text();
      let message = text;
      try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed.detail)) {
          message = parsed.detail.map((d: any) => d.msg).join("; ");
        }
      } catch { /* keep raw */ }
      return {
        content: [{ type: "text" as const, text: `Upload failed (${res.status}): ${message}` }],
        isError: true,
      };
    }

    const result = (await res.json()) as any;
    const fileId = result.file_id ?? result.upload ?? JSON.stringify(result);

    return {
      content: [
        {
          type: "text" as const,
          text:
            `Upload complete.\nFile ID: ${fileId}\n\n` +
            `Pass this ID as 'input' to reducto_parse, reducto_extract, reducto_split, or reducto_classify.`,
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// Tool: reducto_get_job
// ---------------------------------------------------------------------------

server.tool(
  "reducto_get_job",
  "Retrieve the result of a previous Reducto job by its ID. " +
    "Two main uses:\n" +
    "1. Re-use a previous parse without re-processing: pass 'jobid://JOB_ID' as input to " +
    "reducto_extract or reducto_split — but use this tool first to verify the job succeeded.\n" +
    "2. Check the status of a long-running job you kicked off earlier.\n\n" +
    "Returns the full result payload of the original job.",
  {
    job_id: z
      .string()
      .describe(
        "Job ID from a previous reducto_* call. " +
          "Accepts bare UUID ('abc123...') or 'jobid://abc123...' prefix — both work."
      ),
  },
  async (params) => {
    // Normalize: strip jobid:// prefix if present
    const id = params.job_id.replace(/^jobid:\/\//, "");

    const res = await fetch(`${BASE_URL}/job/${id}`, {
      headers: { Authorization: `Bearer ${API_KEY}` },
    });

    if (!res.ok) {
      const text = await res.text();
      let message = text;
      try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed.detail)) {
          message = parsed.detail.map((d: any) => d.msg).join("; ");
        }
      } catch { /* keep raw */ }
      return {
        content: [{ type: "text" as const, text: `Job retrieval failed (${res.status}): ${message}` }],
        isError: true,
      };
    }

    const result = (await res.json()) as any;
    const status = result.status ?? (result.result ? "Completed" : "Processing");

    return {
      content: [
        {
          type: "text" as const,
          text:
            `Job ${id}\nStatus: ${status}\n\n` +
            JSON.stringify(result, null, 2),
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Reducto MCP server v0.2.0 running on stdio");
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
