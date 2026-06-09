/**
 * lib/api.ts
 * ===========
 * Type-safe API client for the copilot backend.
 *
 * v2 additions:
 * - streamQuery: SSE streaming client that yields events as they arrive
 * - ollamaStatus: Check Ollama model availability
 */

import type {
  AnswerResponse,
  QueryRequest,
  HealthStatus,
  ClassificationResult,
  SourceCitation,
} from '@/types/api';

const rawApiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_BASE = rawApiBase.replace(/\/$/, '');

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function apiFetch<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

// ---- SSE Event types ----

export interface StreamStatusEvent {
  type: 'status';
  content: string;
}

export interface StreamPlannerEvent {
  type: 'planner';
  planner: PlannerOutput;
  error: boolean;
}

export interface StreamTokenEvent {
  type: 'token';
  content: string;
}

export interface StreamDoneEvent {
  type: 'done';
  citations: SourceCitation[];
  planner: PlannerOutput;
  tool_mode: string;
  retrieval_count: number;
  latency_ms: number;
  token_count: number;
  follow_up_suggestions: string[];
}

export interface StreamErrorEvent {
  type: 'error';
  content: string;
}

export type StreamEvent =
  | StreamStatusEvent
  | StreamPlannerEvent
  | StreamTokenEvent
  | StreamDoneEvent
  | StreamErrorEvent;

export interface PlannerOutput {
  query_type: string;
  utility_domain: string;
  equipment_tags: string[];
  candidate_tools: string[];
  selected_tool: string;
  confidence: number;
  why_selected: string;
  retrieval_profile: string;
  response_style: string;
}

// ---- API Methods ----

export const api = {
  health(): Promise<HealthStatus> {
    return apiFetch<HealthStatus>('/api/health');
  },

  query(request: QueryRequest): Promise<AnswerResponse> {
    return apiFetch<AnswerResponse>('/api/query', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  classify(query: string): Promise<ClassificationResult> {
    return apiFetch<ClassificationResult>('/api/classify', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  },

  /**
   * Streaming query using Server-Sent Events (SSE).
   * 
   * Calls /api/stream and invokes callbacks for each event type.
   * 
   * Usage:
   *   await api.streamQuery(request, {
   *     onStatus: (msg) => setStatus(msg),
   *     onPlanner: (plan) => setPlanner(plan),
   *     onToken: (token) => setAnswer(prev => prev + token),
   *     onDone: (evt) => setCitations(evt.citations),
   *     onError: (msg) => showError(msg),
   *   });
   */
  async streamQuery(
    request: QueryRequest,
    callbacks: {
      onStatus?: (content: string) => void;
      onPlanner?: (planner: PlannerOutput, error: boolean) => void;
      onToken: (token: string) => void;
      onDone: (event: StreamDoneEvent) => void;
      onError?: (content: string) => void;
    },
  ): Promise<void> {
    const response = await fetch(`${API_BASE}/api/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new ApiError(response.status, await response.text());
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') return;

        try {
          const event: StreamEvent = JSON.parse(raw);
          switch (event.type) {
            case 'status':
              callbacks.onStatus?.(event.content);
              break;
            case 'planner':
              callbacks.onPlanner?.(event.planner, event.error);
              break;
            case 'token':
              callbacks.onToken(event.content);
              break;
            case 'done':
              callbacks.onDone(event);
              break;
            case 'error':
              callbacks.onError?.(event.content);
              break;
          }
        } catch {
          // Skip malformed JSON
        }
      }
    }
  },

  async ollamaStatus(): Promise<{
    available: boolean;
    models: string[];
    current_model: string;
  }> {
    try {
      const resp = await fetch('http://localhost:11434/api/tags', {
        signal: AbortSignal.timeout(2000),
      });
      if (!resp.ok) return { available: false, models: [], current_model: '' };
      const data = await resp.json();
      return {
        available: true,
        models: data.models?.map((m: { name: string }) => m.name) || [],
        current_model: data.models?.[0]?.name || '',
      };
    } catch {
      return { available: false, models: [], current_model: '' };
    }
  },
};
