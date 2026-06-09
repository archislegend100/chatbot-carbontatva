"use client";

import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

type RetrievalMode = "auto" | "fast" | "deep" | "research";

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  
  // Settings State
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("auto");
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Advanced Options
  const [forceColbert, setForceColbert] = useState(false);
  const [forceHyde, setForceHyde] = useState(false);
  const [forceMultiQuery, setForceMultiQuery] = useState(false);
  const [showChunks, setShowChunks] = useState(false);
  const [showScores, setShowScores] = useState(false);
  const [showLatency, setShowLatency] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMsg = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setQuery("");
    setLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const payload = {
        query: userMsg.content,
        retrieval_mode: retrievalMode,
        advanced_options: {
          force_colbert: forceColbert,
          force_hyde: forceHyde,
          force_multi_query: forceMultiQuery,
          show_chunks: showChunks,
          show_scores: showScores,
          show_latency: showLatency
        }
      };

      const res = await fetch(`${apiUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("API Error");

      const data = await res.json();
      setMessages((prev) => [...prev, { role: "assistant", data }]);
    } catch (error) {
      setMessages((prev) => [...prev, { role: "assistant", error: "Failed to connect to the backend API." }]);
    } finally {
      setLoading(false);
    }
  };

  const renderRetrievalPath = (plan: any) => {
    if (!plan) return null;
    const paths = [];
    if (plan.use_dense) paths.push("Semantic Search");
    if (plan.use_sparse) paths.push("Keyword Search");
    if (plan.use_colbert) paths.push("Precision Retrieval");
    if (plan.use_multi_query) paths.push("Query Expansion");
    if (plan.use_hyde) paths.push("Semantic Expansion");
    if (plan.use_reranking) paths.push("Evidence Reordering");
    if (plan.use_parent_expansion) paths.push("Parent Context Expansion");
    if (plan.use_context_compression) paths.push("Context Compression");
    if (plan.use_verification) paths.push("Answer Verification");

    return (
      <div className="mb-6">
        <p className="text-[11px] font-mono text-zinc-500 uppercase tracking-widest mb-3">Execution Path</p>
        <div className="flex flex-wrap gap-2">
          {paths.map(p => (
            <span key={p} className="px-2.5 py-1 bg-zinc-800/50 text-zinc-300 text-[11px] font-medium rounded-sm border border-zinc-700/50">
              {p}
            </span>
          ))}
        </div>
      </div>
    );
  };

  return (
    <main className="min-h-screen bg-[#09090b] text-zinc-300 font-sans flex flex-col items-center selection:bg-zinc-800 selection:text-white">
      {/* Background Subtle Grid - LangChain aesthetic */}
      <div className="fixed inset-0 pointer-events-none bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
      
      {/* Top Header */}
      <header className="w-full max-w-6xl py-6 px-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-6 z-10 sticky top-0 bg-[#09090b]/80 backdrop-blur-xl border-b border-zinc-800/50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-zinc-100 flex items-center justify-center">
            <span className="text-[#09090b] font-bold text-lg leading-none">C</span>
          </div>
          <div>
            <h1 className="text-lg font-medium tracking-tight text-zinc-100">CarbonTatva</h1>
            <p className="text-[11px] font-mono text-zinc-500 tracking-wider">INDUSTRIAL RAG COPILOT</p>
          </div>
        </div>
        
        <div className="flex flex-col items-end gap-3 w-full md:w-auto">
          <div className="flex items-center bg-zinc-900 rounded-md border border-zinc-800 p-1 w-full md:w-auto overflow-x-auto">
            {(["auto", "fast", "deep", "research"] as RetrievalMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setRetrievalMode(mode)}
                className={`px-4 py-1.5 text-[13px] font-medium rounded-sm capitalize transition-all duration-200 ${
                  retrievalMode === mode 
                    ? 'bg-zinc-100 text-zinc-900 shadow-sm' 
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
          <button 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-[11px] font-mono text-zinc-500 hover:text-zinc-300 transition-colors uppercase tracking-widest"
          >
            {showAdvanced ? "[- Hide Developer Settings]" : "[+ Show Developer Settings]"}
          </button>
        </div>
      </header>

      {/* Advanced Settings Dropdown */}
      {showAdvanced && (
        <div className="w-full max-w-6xl bg-zinc-900/50 border-b border-zinc-800/50 px-6 py-8 text-sm grid grid-cols-1 md:grid-cols-2 gap-8 z-10 backdrop-blur-md">
          <div>
            <h3 className="text-[11px] font-mono text-zinc-500 uppercase tracking-widest mb-4">Overrides (Research Only)</h3>
            <div className="space-y-3">
              {[
                { label: "Force Precision Retrieval (ColBERT)", state: forceColbert, setter: setForceColbert },
                { label: "Force Semantic Expansion (HyDE)", state: forceHyde, setter: setForceHyde },
                { label: "Force Query Expansion (Multi-query)", state: forceMultiQuery, setter: setForceMultiQuery },
              ].map((opt, i) => (
                <label key={i} className="flex items-center gap-3 cursor-pointer group">
                  <div className={`w-4 h-4 rounded-sm border flex items-center justify-center transition-colors ${opt.state ? 'bg-zinc-100 border-zinc-100' : 'border-zinc-700 group-hover:border-zinc-500'}`}>
                    {opt.state && <span className="text-[#09090b] text-[10px]">✓</span>}
                  </div>
                  <input type="checkbox" checked={opt.state} onChange={e => opt.setter(e.target.checked)} className="hidden" />
                  <span className="text-[13px] text-zinc-300 group-hover:text-zinc-100 transition-colors">{opt.label}</span>
                </label>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-[11px] font-mono text-zinc-500 uppercase tracking-widest mb-4">Debug Output</h3>
            <div className="space-y-3">
              {[
                { label: "Show Retrieved Chunks", state: showChunks, setter: setShowChunks },
                { label: "Show Retrieval Scores", state: showScores, setter: setShowScores },
                { label: "Show Latency Breakdown", state: showLatency, setter: setShowLatency },
              ].map((opt, i) => (
                <label key={i} className="flex items-center gap-3 cursor-pointer group">
                  <div className={`w-4 h-4 rounded-sm border flex items-center justify-center transition-colors ${opt.state ? 'bg-zinc-100 border-zinc-100' : 'border-zinc-700 group-hover:border-zinc-500'}`}>
                    {opt.state && <span className="text-[#09090b] text-[10px]">✓</span>}
                  </div>
                  <input type="checkbox" checked={opt.state} onChange={e => opt.setter(e.target.checked)} className="hidden" />
                  <span className="text-[13px] text-zinc-300 group-hover:text-zinc-100 transition-colors">{opt.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 w-full max-w-4xl p-6 space-y-10 overflow-y-auto mt-6 pb-40 z-0">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full mt-20 text-center opacity-80">
            <div className="w-16 h-16 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center mb-6 shadow-2xl">
              <span className="text-zinc-100 text-2xl">⚡</span>
            </div>
            <h2 className="text-3xl font-medium tracking-tight text-zinc-100 mb-4">How can I assist?</h2>
            <p className="text-sm text-zinc-500 font-mono tracking-wide max-w-md">
              Ask about boiler efficiency, motor performance, compressed air troubleshooting, or BEE compliance.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`flex flex-col w-full ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
            <div 
              className={`max-w-[85%] rounded-2xl p-6 ${
                msg.role === 'user' 
                  ? 'bg-zinc-100 text-zinc-900 shadow-lg font-medium' 
                  : 'bg-zinc-900/40 border border-zinc-800/50 backdrop-blur-sm'
              }`}
            >
              {msg.role === 'user' ? (
                <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{msg.content}</p>
              ) : (
                msg.error ? (
                  <p className="text-red-400 font-mono text-sm">{msg.error}</p>
                ) : (
                  <div className="space-y-6">
                    
                    {/* Return Debug Metadata Display */}
                    {(retrievalMode === "research" || showLatency || showChunks || showScores) && msg.data.debug && (
                      <div className="bg-[#09090b] p-5 rounded-xl border border-zinc-800 mb-6 font-mono">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                          <h4 className="text-[11px] text-zinc-400 uppercase tracking-widest">System Diagnostics</h4>
                        </div>
                        
                        {msg.data.debug.warnings?.length > 0 && (
                          <div className="mb-5 space-y-2">
                            {msg.data.debug.warnings.map((w: string, i: number) => (
                              <p key={i} className="text-[11px] text-amber-500/90 bg-amber-500/10 p-2.5 rounded-sm border border-amber-500/20">{w}</p>
                            ))}
                          </div>
                        )}

                        {renderRetrievalPath(msg.data.retrieval_plan)}

                        {msg.data.debug.latency_ms && Object.keys(msg.data.debug.latency_ms).length > 0 && (
                          <details className="text-[11px] text-zinc-500 mb-3 group">
                            <summary className="cursor-pointer hover:text-zinc-300 transition-colors list-none flex items-center gap-2">
                              <span className="text-zinc-700 group-open:rotate-90 transition-transform">▶</span>
                              Latency Breakdown (<span className="text-zinc-300">{msg.data.debug.latency_ms.total?.toFixed(0)}ms</span>)
                            </summary>
                            <pre className="mt-3 p-3 bg-zinc-900 rounded-sm border border-zinc-800 overflow-x-auto text-zinc-400">
                              {JSON.stringify(msg.data.debug.latency_ms, null, 2)}
                            </pre>
                          </details>
                        )}

                        {msg.data.debug.scores?.length > 0 && (
                          <details className="text-[11px] text-zinc-500 mb-3 group">
                            <summary className="cursor-pointer hover:text-zinc-300 transition-colors list-none flex items-center gap-2">
                              <span className="text-zinc-700 group-open:rotate-90 transition-transform">▶</span>
                              Retrieval Scores
                            </summary>
                            <pre className="mt-3 p-3 bg-zinc-900 rounded-sm border border-zinc-800 overflow-x-auto text-zinc-400">
                              {JSON.stringify(msg.data.debug.scores, null, 2)}
                            </pre>
                          </details>
                        )}
                        
                        {msg.data.debug.retrieved_chunks?.length > 0 && (
                          <details className="text-[11px] text-zinc-500 group">
                            <summary className="cursor-pointer hover:text-zinc-300 transition-colors list-none flex items-center gap-2">
                              <span className="text-zinc-700 group-open:rotate-90 transition-transform">▶</span>
                              Retrieved Chunks ({msg.data.debug.retrieved_chunks.length})
                            </summary>
                            <div className="mt-3 space-y-3">
                              {msg.data.debug.retrieved_chunks.map((chunk: any, i: number) => (
                                <div key={i} className="bg-zinc-900 p-3 rounded-sm border border-zinc-800 overflow-x-auto">
                                  <p className="text-[9px] text-zinc-500 mb-2">ID: {chunk.chunk_id}</p>
                                  <p className="text-[11px] text-zinc-400 whitespace-pre-wrap leading-relaxed">{chunk.text}</p>
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    )}

                    {/* Answer Area */}
                    <div className="prose prose-invert prose-zinc max-w-none prose-p:leading-relaxed prose-pre:bg-zinc-900 prose-pre:border prose-pre:border-zinc-800">
                      <ReactMarkdown>{msg.data.answer}</ReactMarkdown>
                    </div>
                    
                    {/* Citations Area */}
                    {msg.data.citations && msg.data.citations.length > 0 && (
                      <div className="pt-6 mt-4 border-t border-zinc-800/50">
                        <h4 className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest mb-4">Grounding Sources</h4>
                        <div className="flex flex-wrap gap-3">
                          {msg.data.citations.map((cit: any, i: number) => (
                            <details key={i} className="group relative text-sm bg-zinc-900 border border-zinc-800 hover:border-zinc-700 transition-colors rounded-lg p-3 max-w-full lg:max-w-[48%] flex-1">
                              <summary className="cursor-pointer font-medium text-zinc-300 list-none flex items-center gap-2">
                                <span className="flex-shrink-0 w-5 h-5 rounded bg-zinc-800 text-zinc-400 flex items-center justify-center text-[10px]">{i+1}</span>
                                <span className="truncate">{cit.book}</span>
                                {cit.page > 0 && <span className="text-zinc-500 text-xs shrink-0 bg-zinc-950 px-1.5 py-0.5 rounded">pg {cit.page}</span>}
                              </summary>
                              <div className="mt-4 text-zinc-400 font-mono text-[11px] bg-[#09090b] p-4 rounded border border-zinc-800 whitespace-pre-wrap leading-relaxed overflow-x-auto">
                                {cit.text_snippet}
                              </div>
                            </details>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex items-start w-full">
            <div className="bg-zinc-900/40 border border-zinc-800/50 backdrop-blur-sm rounded-2xl p-6 shadow-sm flex items-center space-x-3">
              <div className="flex space-x-1">
                <div className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce"></div>
                <div className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce delay-150"></div>
                <div className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce delay-300"></div>
              </div>
              <span className="text-[13px] font-mono text-zinc-400 tracking-wide uppercase">Processing...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="w-full max-w-4xl fixed bottom-8 z-20 px-6">
        <form onSubmit={handleSubmit} className="relative flex items-center shadow-2xl">
          <input
            type="text"
            className="w-full bg-zinc-900/80 backdrop-blur-xl border border-zinc-800 text-zinc-100 placeholder:text-zinc-500 rounded-2xl pl-6 pr-32 py-5 focus:outline-none focus:ring-1 focus:ring-zinc-600 focus:border-zinc-600 transition-all text-[15px]"
            placeholder="Query the engineering knowledge base..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="absolute right-2 top-2 bottom-2 bg-zinc-100 text-[#09090b] px-6 rounded-xl font-semibold hover:bg-white transition-all disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98] text-[14px]"
          >
            Send
          </button>
        </form>
      </div>
    </main>
  );
}
