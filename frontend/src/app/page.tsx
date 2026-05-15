'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import type { ToolMode, UtilityDomain, ExplanationLevel, SourceCitation } from '@/types/api';
import { api, ApiError, type PlannerOutput, type StreamDoneEvent } from '@/lib/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// ============================================================
// CONSTANTS
// ============================================================

const TOOL_MODES: { id: ToolMode; label: string; icon: string; description: string }[] = [
  { id: 'auto',        label: 'Auto',         icon: '*',  description: 'AI chooses the best mode' },
  { id: 'qa',          label: 'Q&A',          icon: 'Q',  description: 'Direct technical answers' },
  { id: 'explainer',   label: 'Explain',      icon: 'E',  description: 'Deep concept explanations' },
  { id: 'troubleshoot',label: 'Troubleshoot', icon: 'T',  description: 'Diagnose energy problems' },
  { id: 'opportunity', label: 'Opportunities',icon: 'O',  description: 'Find energy savings' },
  { id: 'comparison',  label: 'Compare',      icon: 'C',  description: 'Compare systems' },
  { id: 'navigation',  label: 'Navigate',     icon: 'N',  description: 'Find chapters & sections' },
  { id: 'checklist',   label: 'Checklist',    icon: 'CL', description: 'Audit checklists' },
  { id: 'summarize',   label: 'Summarize',    icon: 'S',  description: 'Chapter summaries' },
];

const SAMPLE_QUERIES = [
  "What is the optimum excess air percentage for a coal-fired boiler?",
  "How does a VFD reduce energy consumption in motors?",
  "Compare fire tube and water tube boilers",
  "Generate an energy audit checklist for compressed air systems",
  "Why is my boiler efficiency dropping below 75%?",
  "Explain power factor and its industrial importance",
  "What are the top energy saving opportunities in steam systems?",
  "Summarize waste heat recovery methods",
];

function generateId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

// ============================================================
// TYPES
// ============================================================

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  status?: string;           // Live status during streaming
  toolMode?: string;
  domain?: string;
  citations?: SourceCitation[];
  followUps?: string[];
  planner?: PlannerOutput;
  plannerError?: boolean;
  latencyMs?: number;
  tokenCount?: number;
}

// ============================================================
// MAIN PAGE
// ============================================================

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [toolMode, setToolMode] = useState<ToolMode>('auto');
  const [domain, setDomain] = useState<UtilityDomain | ''>('');
  const [explanationLevel, setExplanationLevel] = useState<ExplanationLevel>('engineer');
  const [indexLoaded, setIndexLoaded] = useState<boolean | null>(null);
  const [ollamaStatus, setOllamaStatus] = useState<{ available: boolean; model: string } | null>(null);
  const [selectedCitations, setSelectedCitations] = useState<SourceCitation[]>([]);
  const [showEvidence, setShowEvidence] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [debugPlanner, setDebugPlanner] = useState<PlannerOutput | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Check backend + Ollama on mount
  useEffect(() => {
    (async () => {
      try {
        const health = await api.health();
        setIndexLoaded(health.index_loaded);
      } catch {
        setIndexLoaded(false);
      }
    })();
    (async () => {
      const status = await api.ollamaStatus();
      setOllamaStatus({ available: status.available, model: status.current_model });
    })();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const ta = e.target;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  };

  // ---- STREAMING SEND ----
  const sendMessage = useCallback(async (queryText?: string) => {
    const query = (queryText ?? input).trim();
    if (!query || isLoading) return;

    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = '56px';

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: query,
      timestamp: new Date(),
    };

    const assistantId = generateId();
    const assistantPlaceholder: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
      status: 'Connecting...',
    };

    setMessages(prev => [...prev, userMsg, assistantPlaceholder]);
    setIsLoading(true);

    let fullText = '';

    try {
      await api.streamQuery(
        {
          query,
          tool_mode: toolMode,
          domain_filter: domain || null,
          explanation_level: explanationLevel,
        },
        {
          onStatus: (content) => {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, status: content }
                  : m,
              ),
            );
          },

          onPlanner: (planner, error) => {
            setDebugPlanner(planner);
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? {
                      ...m,
                      toolMode: planner.selected_tool,
                      domain: planner.utility_domain,
                      planner,
                      plannerError: error,
                      status: 'Generating answer...',
                    }
                  : m,
              ),
            );
          },

          onToken: (token) => {
            fullText += token;
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, content: fullText, status: undefined }
                  : m,
              ),
            );
          },

          onDone: (evt: StreamDoneEvent) => {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: fullText,
                      isStreaming: false,
                      status: undefined,
                      toolMode: evt.tool_mode,
                      citations: evt.citations,
                      followUps: evt.follow_up_suggestions,
                      planner: evt.planner,
                      latencyMs: evt.latency_ms,
                      tokenCount: evt.token_count,
                    }
                  : m,
              ),
            );
            if (evt.citations.length > 0) {
              setSelectedCitations(evt.citations);
              setShowEvidence(true);
            }
          },

          onError: (content) => {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, role: 'error' as const, content, isStreaming: false, status: undefined }
                  : m,
              ),
            );
          },
        },
      );
    } catch (err) {
      const errorMsg = err instanceof ApiError
        ? err.message
        : 'Failed to connect to backend. Make sure the server is running.';
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId
            ? { ...m, role: 'error' as const, content: errorMsg, isStreaming: false, status: undefined }
            : m,
        ),
      );
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, toolMode, domain, explanationLevel]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSelectedCitations([]);
    setShowEvidence(false);
    setDebugPlanner(null);
  };

  const domainLabel = domain === 'thermal' ? 'Thermal' : domain === 'electrical' ? 'Electrical' : 'All Domains';

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg-base)' }}>

      {/* ========== SIDEBAR ========== */}
      {sidebarOpen && (
        <aside style={{
          width: 'var(--sidebar-width)',
          minWidth: 'var(--sidebar-width)',
          background: 'var(--bg-surface)',
          borderRight: '1px solid var(--border-subtle)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}>
          {/* Logo */}
          <div style={{ padding: '20px 20px 16px', borderBottom: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <div style={{
                width: 36, height: 36,
                background: 'linear-gradient(135deg, #00d4aa, #00b891)',
                borderRadius: 10,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 18,
                boxShadow: '0 0 12px var(--accent-primary-glow)',
                fontWeight: 700, color: '#0a1018',
              }}>E</div>
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--text-primary)' }}>
                  Energy Copilot
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                  LangGraph · Mistral · v2.1
                </div>
              </div>
            </div>

            {/* Ollama status badge */}
            <div style={{
              marginTop: 10, padding: '5px 10px',
              borderRadius: 6,
              background: ollamaStatus?.available ? 'var(--accent-primary-dim)' : '#f59e0b11',
              border: `1px solid ${ollamaStatus?.available ? 'var(--accent-primary)30' : '#f59e0b33'}`,
              fontSize: '0.68rem',
              color: ollamaStatus?.available ? 'var(--accent-primary)' : 'var(--warning)',
              display: 'flex', alignItems: 'center', gap: 5,
            }}>
              <div style={{
                width: 6, height: 6, borderRadius: '50%',
                background: ollamaStatus?.available ? 'var(--accent-primary)' : 'var(--warning)',
                boxShadow: ollamaStatus?.available ? '0 0 4px var(--accent-primary)' : 'none',
              }} />
              {ollamaStatus === null
                ? 'Checking LLM...'
                : ollamaStatus.available
                  ? `LLM Ready (${ollamaStatus.model || 'connected'})`
                  : 'LLM unavailable — check .env'
              }
            </div>
          </div>

          {/* Domain Filter */}
          <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Domain
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {[
                { value: '', label: 'All Domains', desc: 'Both manuals' },
                { value: 'thermal', label: 'Thermal', desc: 'Boilers, furnaces, steam' },
                { value: 'electrical', label: 'Electrical', desc: 'Motors, lighting, tariffs' },
              ].map(opt => (
                <button
                  key={opt.value}
                  onClick={() => setDomain(opt.value as UtilityDomain | '')}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '7px 10px', borderRadius: 7,
                    background: domain === opt.value ? 'var(--bg-elevated)' : 'transparent',
                    border: domain === opt.value ? '1px solid var(--border-accent)' : '1px solid transparent',
                    cursor: 'pointer', width: '100%', textAlign: 'left',
                  }}
                >
                  <div>
                    <div style={{ fontSize: '0.83rem', color: domain === opt.value ? 'var(--accent-primary)' : 'var(--text-secondary)', fontWeight: 500 }}>
                      {opt.label}
                    </div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{opt.desc}</div>
                  </div>
                  {domain === opt.value && <span style={{ color: 'var(--accent-primary)', fontSize: '0.7rem' }}>✓</span>}
                </button>
              ))}
            </div>
          </div>

          {/* Tool Modes */}
          <div style={{ padding: '14px 20px', flex: 1, overflowY: 'auto' }}>
            <div style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Tool Mode
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {TOOL_MODES.map(mode => (
                <button
                  key={mode.id}
                  onClick={() => setToolMode(mode.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '7px 10px', borderRadius: 7,
                    border: toolMode === mode.id ? '1px solid var(--border-accent)' : '1px solid transparent',
                    background: toolMode === mode.id ? 'var(--accent-primary-dim)' : 'transparent',
                    cursor: 'pointer', width: '100%', textAlign: 'left',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <span style={{ fontSize: '0.95rem', width: 20, textAlign: 'center' }}>{mode.icon}</span>
                  <div>
                    <div style={{ fontSize: '0.82rem', color: toolMode === mode.id ? 'var(--accent-primary)' : 'var(--text-secondary)', fontWeight: toolMode === mode.id ? 600 : 400 }}>
                      {mode.label}
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{mode.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Response Style + Status */}
          <div style={{ padding: '10px 20px', borderTop: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Style
            </div>
            <div style={{ display: 'flex', gap: 5, marginBottom: 10 }}>
              {(['beginner', 'engineer'] as ExplanationLevel[]).map(level => (
                <button
                  key={level}
                  onClick={() => setExplanationLevel(level)}
                  style={{
                    flex: 1, padding: '5px',
                    borderRadius: 6,
                    border: explanationLevel === level ? '1px solid var(--accent-primary)' : '1px solid var(--border-dim)',
                    background: explanationLevel === level ? 'var(--accent-primary-dim)' : 'transparent',
                    cursor: 'pointer', fontSize: '0.74rem',
                    color: explanationLevel === level ? 'var(--accent-primary)' : 'var(--text-secondary)',
                    fontWeight: 500, textTransform: 'capitalize',
                  }}
                >
                  {level === 'beginner' ? 'Beginner' : 'Engineer'}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                <div style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: indexLoaded === null ? 'var(--warning)' : indexLoaded ? 'var(--success)' : 'var(--error)',
                  boxShadow: indexLoaded ? '0 0 5px var(--success)' : 'none',
                }} />
                {indexLoaded === null ? 'Checking...' : indexLoaded ? 'KB Ready' : 'No Index'}
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  onClick={() => setShowDebug(d => !d)}
                  style={{
                    fontSize: '0.65rem', color: showDebug ? 'var(--accent-primary)' : 'var(--text-muted)',
                    background: 'none', border: 'none', cursor: 'pointer',
                    padding: '2px 6px', borderRadius: 4,
                  }}
                >
                  {showDebug ? 'Debug ON' : 'Debug'}
                </button>
                {messages.length > 0 && (
                  <button onClick={clearChat} style={{ fontSize: '0.65rem', color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer', padding: '2px 4px', borderRadius: 4 }}>
                    Clear
                  </button>
                )}
              </div>
            </div>
          </div>
        </aside>
      )}

      {/* ========== MAIN CHAT ========== */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

        {/* Header */}
        <header style={{
          height: 'var(--header-height)',
          display: 'flex', alignItems: 'center', padding: '0 20px',
          borderBottom: '1px solid var(--border-subtle)',
          background: 'var(--bg-surface)', gap: 12, flexShrink: 0,
        }}>
          <button onClick={() => setSidebarOpen(s => !s)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 18, padding: 4 }}>
            ☰
          </button>
          <div style={{ flex: 1, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            <span style={{ color: toolMode === 'auto' ? 'var(--accent-primary)' : 'var(--text-secondary)', fontWeight: 600 }}>
              {TOOL_MODES.find(m => m.id === toolMode)?.label}
            </span>
            {' · '}
            <span style={{ color: domain === 'thermal' ? 'var(--thermal-primary)' : domain === 'electrical' ? 'var(--electrical-primary)' : 'var(--text-muted)' }}>
              {domainLabel}
            </span>
            {' · '}
            <span>{explanationLevel === 'beginner' ? 'Beginner' : 'Engineer'}</span>
            {' · '}
            <span style={{ color: 'var(--accent-primary)', fontWeight: 500 }}>LangGraph Pipeline</span>
          </div>

          {selectedCitations.length > 0 && (
            <button className="btn btn-secondary btn-sm" onClick={() => setShowEvidence(e => !e)}>
              Evidence ({selectedCitations.length})
            </button>
          )}
        </header>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 20 }}>
          {messages.length === 0 && (
            <WelcomeScreen indexLoaded={indexLoaded} llmAvailable={ollamaStatus?.available ?? null} onQuery={sendMessage} />
          )}
          {messages.map(msg => (
            <MessageBubble
              key={msg.id}
              message={msg}
              showDebug={showDebug}
              onFollowUp={(q) => sendMessage(q)}
              onShowEvidence={(cits) => {
                setSelectedCitations(cits);
                setShowEvidence(true);
              }}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div style={{ padding: '14px 24px 18px', borderTop: '1px solid var(--border-subtle)', background: 'var(--bg-surface)', flexShrink: 0 }}>
          {!indexLoaded && indexLoaded !== null && (
            <div style={{
              background: '#f59e0b11', border: '1px solid #f59e0b33', borderRadius: 10,
              padding: '9px 14px', marginBottom: 10, fontSize: '0.78rem', color: 'var(--warning)',
            }}>
              Knowledge base not indexed. Run:{' '}
              <code style={{ background: 'var(--bg-elevated)', padding: '1px 6px', borderRadius: 4, color: 'var(--accent-primary)' }}>
                ./run_ingest.sh
              </code>
            </div>
          )}
          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
            <textarea
              ref={textareaRef}
              className="input"
              placeholder={`Ask about energy efficiency... (${domainLabel})`}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={1}
              style={{ minHeight: 52, maxHeight: 160, resize: 'none', flex: 1, paddingTop: 14, paddingBottom: 14, fontSize: '0.93rem' }}
            />
            <button
              className="btn btn-primary"
              onClick={() => sendMessage()}
              disabled={isLoading || !input.trim()}
              style={{ height: 52, paddingLeft: 20, paddingRight: 20, flexShrink: 0 }}
            >
              {isLoading ? <div className="spinner" style={{ width: 15, height: 15 }} /> : '→'}
            </button>
          </div>
          <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)', marginTop: 6, textAlign: 'center' }}>
            Grounded in BEE Thermal &amp; Electrical Manuals · LangGraph + Mistral
          </div>
        </div>
      </main>

      {/* ========== EVIDENCE PANEL ========== */}
      {showEvidence && selectedCitations.length > 0 && (
        <aside className="evidence-panel animate-slide-in" style={{
          width: 320, minWidth: 320,
          background: 'var(--bg-surface)',
          borderLeft: '1px solid var(--border-subtle)',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-primary)' }}>Source Evidence</div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{selectedCitations.length} retrieved + reranked chunks</div>
            </div>
            <button onClick={() => setShowEvidence(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 18 }}>×</button>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {selectedCitations.map((citation, i) => (
              <CitationCard key={citation.chunk_id} citation={citation} rank={i + 1} />
            ))}
          </div>
        </aside>
      )}
    </div>
  );
}

// ============================================================
// WELCOME SCREEN
// ============================================================

function WelcomeScreen({ indexLoaded, llmAvailable, onQuery }: {
  indexLoaded: boolean | null;
  llmAvailable: boolean | null;
  onQuery: (q: string) => void;
}) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px 20px', maxWidth: 740, margin: '0 auto', width: '100%' }}>
      <div style={{ textAlign: 'center', marginBottom: 44 }} className="animate-fade-in-up">
        <div style={{ fontSize: 54, marginBottom: 14, filter: 'drop-shadow(0 0 20px #00d4aa66)', fontWeight: 700, color: 'var(--accent-primary)' }}>E</div>
        <h1 style={{ fontSize: '1.9rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: 10, letterSpacing: '-0.02em' }}>
          Industrial Energy Efficiency Copilot
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', maxWidth: 500, margin: '0 auto', lineHeight: 1.7 }}>
          Grounded intelligence from BEE Thermal &amp; Electrical Manuals.
          Powered by <strong style={{ color: 'var(--accent-primary)' }}>LangGraph</strong> orchestration and{' '}
          <strong style={{ color: 'var(--accent-primary)' }}>Mistral</strong> cloud inference.
        </p>
        <div style={{ marginTop: 18, display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
          {['Thermal', 'Electrical', '8 Specialist Tools', 'Cloud-Native'].map(tag => (
            <span key={tag} className="badge badge-accent" style={{ fontSize: '0.72rem', padding: '4px 10px' }}>{tag}</span>
          ))}
        </div>

        {/* Status row */}
        <div style={{ marginTop: 16, display: 'flex', gap: 10, justifyContent: 'center', flexWrap: 'wrap' }}>
          <div style={{
            padding: '4px 12px', borderRadius: 20, fontSize: '0.7rem', fontWeight: 500,
            background: llmAvailable === null ? 'var(--bg-elevated)' : llmAvailable ? 'var(--accent-primary-dim)' : '#f59e0b11',
            color: llmAvailable === null ? 'var(--text-muted)' : llmAvailable ? 'var(--accent-primary)' : 'var(--warning)',
            border: `1px solid ${llmAvailable ? 'var(--accent-primary)30' : '#f59e0b33'}`,
          }}>
            {llmAvailable === null ? 'Checking LLM...' : llmAvailable ? 'LLM Ready' : 'LLM Unavailable'}
          </div>
          <div style={{
            padding: '4px 12px', borderRadius: 20, fontSize: '0.7rem', fontWeight: 500,
            background: indexLoaded === null ? 'var(--bg-elevated)' : indexLoaded ? 'var(--accent-primary-dim)' : '#ef444411',
            color: indexLoaded === null ? 'var(--text-muted)' : indexLoaded ? 'var(--accent-primary)' : '#f87171',
            border: `1px solid ${indexLoaded ? 'var(--accent-primary)30' : '#ef444433'}`,
          }}>
            {indexLoaded === null ? 'Checking index...' : indexLoaded ? 'KB Indexed' : 'Run Ingestion'}
          </div>
        </div>
      </div>

      {/* Sample queries */}
      <div style={{ width: '100%' }}>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'center', marginBottom: 14, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Try these queries
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {SAMPLE_QUERIES.map((q, i) => (
            <button
              key={i}
              onClick={() => onQuery(q)}
              className="glass-card"
              style={{
                padding: '11px 13px', cursor: 'pointer', textAlign: 'left',
                border: '1px solid var(--border-dim)', width: '100%',
                background: 'var(--bg-card)', borderRadius: 10,
                transition: 'all 0.2s ease', fontSize: '0.8rem',
                color: 'var(--text-secondary)', lineHeight: 1.5,
              }}
              onMouseEnter={e => {
                (e.currentTarget).style.borderColor = 'var(--accent-primary)';
                (e.currentTarget).style.color = 'var(--text-primary)';
              }}
              onMouseLeave={e => {
                (e.currentTarget).style.borderColor = 'var(--border-dim)';
                (e.currentTarget).style.color = 'var(--text-secondary)';
              }}
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================================
// MESSAGE BUBBLE
// ============================================================

function MessageBubble({ message, showDebug, onFollowUp, onShowEvidence }: {
  message: ChatMessage;
  showDebug: boolean;
  onFollowUp: (q: string) => void;
  onShowEvidence: (citations: SourceCitation[]) => void;
}) {
  const isUser = message.role === 'user';
  const isError = message.role === 'error';
  const [plannerOpen, setPlannerOpen] = useState(false);

  if (isUser) {
    return (
      <div className="animate-fade-in-up" style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <div style={{
          maxWidth: '70%',
          background: 'linear-gradient(135deg, var(--accent-primary), #00b891)',
          borderRadius: '16px 16px 4px 16px',
          padding: '12px 16px', color: '#0a1018',
          fontWeight: 500, fontSize: '0.9rem', lineHeight: 1.6,
          boxShadow: '0 4px 16px var(--accent-primary-glow)',
        }}>
          {message.content}
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="animate-fade-in-up" style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
        <div style={{ width: 34, height: 34, borderRadius: 10, background: '#ef444422', border: '1px solid #ef444444', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: 12, fontWeight: 700, color: '#f87171' }}>!</div>
        <div style={{ background: '#ef444411', border: '1px solid #ef444433', borderRadius: '4px 14px 14px 14px', padding: '11px 15px', color: '#f87171', fontSize: '0.86rem', lineHeight: 1.6 }}>
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in-up" style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
      {/* Avatar */}
      <div style={{
        width: 34, height: 34, borderRadius: 10,
        background: 'linear-gradient(135deg, var(--bg-card), var(--bg-elevated))',
        border: '1px solid var(--border-dim)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0, fontSize: 16,
      }}>
        {TOOL_MODES.find(m => m.id === message.toolMode)?.icon ?? '*'}
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Badges */}
        {(message.toolMode || message.isStreaming) && (
          <div style={{ display: 'flex', gap: 5, marginBottom: 7, flexWrap: 'wrap', alignItems: 'center' }}>
            {message.toolMode && (
              <span className="badge badge-accent" style={{ fontSize: '0.65rem' }}>
                {TOOL_MODES.find(m => m.id === message.toolMode)?.label ?? message.toolMode}
              </span>
            )}
            {message.domain && message.domain !== 'both' && message.domain !== 'unknown' && (
              <span className={`badge badge-${message.domain}`} style={{ fontSize: '0.65rem' }}>
                {message.domain === 'thermal' ? 'Thermal' : 'Electrical'}
              </span>
            )}
            {message.latencyMs && (
              <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>
                {(message.latencyMs / 1000).toFixed(1)}s
              </span>
            )}
            {message.tokenCount && (
              <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>
                {message.tokenCount} tokens
              </span>
            )}
            {message.isStreaming && message.status && (
              <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                {message.status}
              </span>
            )}
          </div>
        )}

        {/* Content */}
        <div className="glass-card" style={{ padding: '14px 18px', borderRadius: '4px 14px 14px 14px', background: 'var(--bg-card)' }}>
          {message.isStreaming && message.content === '' ? (
            <div>
              <div className="typing-dots">
                <div className="typing-dot" />
                <div className="typing-dot" />
                <div className="typing-dot" />
              </div>
              {message.status && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 6 }}>
                  {message.status}
                </div>
              )}
            </div>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content || ''}
              </ReactMarkdown>
              {message.isStreaming && (
                <span style={{
                  display: 'inline-block', width: 2, height: '1em',
                  background: 'var(--accent-primary)', marginLeft: 2,
                  animation: 'blink 1s infinite',
                  verticalAlign: 'text-bottom',
                }} />
              )}
            </div>
          )}
        </div>

        {/* Action buttons */}
        {!message.isStreaming && (
          <div style={{ marginTop: 8, display: 'flex', gap: 5, flexWrap: 'wrap', alignItems: 'center' }}>
            {/* Evidence button removed — sources accessible via header panel */}
            {showDebug && message.planner && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setPlannerOpen(p => !p)}
                style={{ fontSize: '0.7rem', color: 'var(--accent-primary)' }}
              >
                Planner {plannerOpen ? '▲' : '▼'}
              </button>
            )}
            {message.followUps?.slice(0, 2).map((qu, i) => (
              <button
                key={i}
                className="btn btn-ghost btn-sm"
                onClick={() => onFollowUp(qu)}
                style={{ fontSize: '0.72rem', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                title={qu}
              >
                → {qu}
              </button>
            ))}
          </div>
        )}

        {/* Debug: Planner reasoning panel */}
        {showDebug && plannerOpen && message.planner && (
          <div style={{
            marginTop: 8, padding: '12px 14px',
            background: 'var(--bg-elevated)', border: '1px solid var(--border-accent)',
            borderRadius: 10, fontSize: '0.75rem',
          }}>
            <div style={{ fontWeight: 600, color: 'var(--accent-primary)', marginBottom: 8 }}>
              LangGraph Planner {message.plannerError && '(fallback)'}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px' }}>
              {[
                ['Tool', message.planner.selected_tool],
                ['Domain', message.planner.utility_domain],
                ['Confidence', `${((message.planner.confidence || 0) * 100).toFixed(0)}%`],
                ['Style', message.planner.response_style],
                ['Retrieval', message.planner.retrieval_profile],
              ].map(([label, val]) => (
                <div key={label} style={{ color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{label}: </span>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{val}</span>
                </div>
              ))}
            </div>
            {message.planner.why_selected && (
              <div style={{ marginTop: 6, color: 'var(--text-secondary)', fontStyle: 'italic', borderTop: '1px solid var(--border-subtle)', paddingTop: 6 }}>
                {message.planner.why_selected}
              </div>
            )}
            {message.planner.equipment_tags?.length > 0 && (
              <div style={{ marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {message.planner.equipment_tags.map(tag => (
                  <span key={tag} style={{ background: 'var(--accent-primary-dim)', color: 'var(--accent-primary)', padding: '1px 6px', borderRadius: 4, fontSize: '0.65rem' }}>
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================
// CITATION CARD
// ============================================================

function CitationCard({ citation, rank }: { citation: SourceCitation; rank: number }) {
  const [expanded, setExpanded] = useState(false);
  const isThermal = citation.utility_domain === 'thermal';

  return (
    <div
      className="glass-card"
      style={{
        padding: '11px 13px', borderRadius: 10,
        borderLeft: `3px solid ${isThermal ? 'var(--thermal-primary)' : 'var(--electrical-primary)'}`,
        cursor: 'pointer',
      }}
      onClick={() => setExpanded(e => !e)}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 3 }}>
            <span style={{
              width: 17, height: 17, borderRadius: 4,
              background: isThermal ? 'var(--thermal-dim)' : 'var(--electrical-dim)',
              color: isThermal ? 'var(--thermal-primary)' : 'var(--electrical-primary)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.62rem', fontWeight: 700, flexShrink: 0,
            }}>{rank}</span>
            <span style={{ fontSize: '0.67rem', color: 'var(--text-muted)', fontWeight: 500 }}>
              {citation.book_name?.includes('Thermal') ? 'BEE Thermal' : 'BEE Electrical'}
            </span>
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-primary)', fontWeight: 500, lineHeight: 1.3, marginBottom: 3 }}>
            {citation.chapter_title ?? 'Unknown Chapter'}
          </div>
          {citation.section_title && (
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>§ {citation.section_title}</div>
          )}
          <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)', marginTop: 3, display: 'flex', gap: 8 }}>
            <span>pp. {citation.page_start}–{citation.page_end}</span>
            <span>Score: {(citation.relevance_score * 100).toFixed(0)}%</span>
          </div>
        </div>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem', flexShrink: 0 }}>
          {expanded ? '▲' : '▼'}
        </span>
      </div>

      {expanded && (
        <div style={{
          borderTop: '1px solid var(--border-subtle)',
          fontSize: '0.76rem', color: 'var(--text-secondary)',
          lineHeight: 1.6, fontFamily: 'var(--font-mono)',
          background: 'var(--bg-base)', padding: '8px 10px',
          borderRadius: 6, marginTop: 10,
        }}>
          {citation.snippet}
        </div>
      )}
    </div>
  );
}
