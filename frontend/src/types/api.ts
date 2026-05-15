/**
 * types/api.ts
 * TypeScript types matching the backend Pydantic schemas.
 */

export type UtilityDomain = 'thermal' | 'electrical' | 'unknown';

export type ChunkType = 'section' | 'semantic' | 'table' | 'list' | 'formula' | 'figure';

export type ToolMode =
  | 'qa'
  | 'explainer'
  | 'troubleshoot'
  | 'opportunity'
  | 'comparison'
  | 'navigation'
  | 'checklist'
  | 'summarize'
  | 'auto';

export type ExplanationLevel = 'beginner' | 'engineer';

export interface QueryRequest {
  query: string;
  tool_mode: ToolMode;
  domain_filter?: UtilityDomain | null;
  explanation_level: ExplanationLevel;
  top_k?: number;
}

export interface SourceCitation {
  chunk_id: string;
  book_name: string;
  utility_domain: string;
  chapter_title?: string;
  section_title?: string;
  page_start: number;
  page_end: number;
  relevance_score: number;
  snippet: string;
}

export interface ClassificationResult {
  tool_mode: ToolMode;
  utility_domain?: UtilityDomain;
  equipment_tags: string[];
  concept_tags: string[];
  confidence: number;
  reasoning?: string;
}

export interface AnswerResponse {
  query: string;
  tool_mode: ToolMode;
  answer: string;
  structured_sections: Record<string, unknown>;
  citations: SourceCitation[];
  classification?: ClassificationResult;
  follow_up_suggestions: string[];
  retrieval_count: number;
  generation_model: string;
  latency_ms?: number;
}

export interface HealthStatus {
  status: string;
  version: string;
  index_loaded: boolean;
  embedding_model: string;
  llm_provider: string;
  chunk_count: number;
}

// ---- Chat Message (frontend-only) ----

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  timestamp: Date;
  toolMode?: ToolMode;
  domain?: UtilityDomain;
  citations?: SourceCitation[];
  followUps?: string[];
  classification?: ClassificationResult;
  latencyMs?: number;
  isStreaming?: boolean;
}
