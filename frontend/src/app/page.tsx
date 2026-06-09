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
      <div className="mb-4">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Search Path Used:</p>
        <div className="flex flex-wrap gap-2">
          {paths.map(p => (
            <span key={p} className="px-2 py-1 bg-slate-100 text-slate-600 text-[11px] rounded-md border border-slate-200">
              {p}
            </span>
          ))}
        </div>
      </div>
    );
  };

  return (
    <main className="min-h-screen bg-[#fafafa] text-slate-900 font-sans flex flex-col items-center">
      <header className="w-full max-w-5xl bg-white border-b border-slate-200 py-5 px-8 flex justify-between items-start md:items-center flex-col md:flex-row gap-4 shadow-sm z-10 sticky top-0">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-800">CarbonTatvaAI</h1>
          <p className="text-sm text-slate-500 mt-1">Answers grounded in indexed BEE thermal and electrical utility manuals.</p>
        </div>
        
        <div className="flex flex-col items-end gap-2 w-full md:w-auto">
          <div className="flex items-center gap-2 bg-slate-50 p-1.5 rounded-lg border border-slate-200 w-full md:w-auto justify-between md:justify-start">
            <span className="text-xs font-medium text-slate-500 ml-2">Mode:</span>
            <div className="flex gap-1">
              {(["auto", "fast", "deep", "research"] as RetrievalMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setRetrievalMode(mode)}
                  className={`px-3 py-1 text-xs font-medium rounded-md capitalize transition-colors ${
                    retrievalMode === mode 
                      ? 'bg-white text-blue-700 shadow-sm border border-slate-200' 
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>
          <button 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs text-blue-600 hover:text-blue-800 underline decoration-blue-200 underline-offset-4"
          >
            {showAdvanced ? "Hide Advanced Settings" : "Show Advanced Settings"}
          </button>
        </div>
      </header>

      {showAdvanced && (
        <div className="w-full max-w-5xl bg-white border-b border-slate-200 px-8 py-5 text-sm grid grid-cols-1 md:grid-cols-2 gap-6 shadow-inner">
          <div>
            <h3 className="font-semibold text-slate-700 mb-3">Developer Overrides (Research Mode Only)</h3>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={forceColbert} onChange={e => setForceColbert(e.target.checked)} className="rounded text-blue-600 focus:ring-blue-500" />
                <span className="text-slate-600">Force Precision Retrieval (ColBERT)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={forceHyde} onChange={e => setForceHyde(e.target.checked)} className="rounded text-blue-600 focus:ring-blue-500" />
                <span className="text-slate-600">Force Semantic Expansion (HyDE)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={forceMultiQuery} onChange={e => setForceMultiQuery(e.target.checked)} className="rounded text-blue-600 focus:ring-blue-500" />
                <span className="text-slate-600">Force Query Expansion (Multi-query)</span>
              </label>
            </div>
          </div>
          <div>
            <h3 className="font-semibold text-slate-700 mb-3">Debug Output Controls</h3>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={showChunks} onChange={e => setShowChunks(e.target.checked)} className="rounded text-blue-600 focus:ring-blue-500" />
                <span className="text-slate-600">Show Retrieved Chunks</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={showScores} onChange={e => setShowScores(e.target.checked)} className="rounded text-blue-600 focus:ring-blue-500" />
                <span className="text-slate-600">Show Retrieval Scores</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={showLatency} onChange={e => setShowLatency(e.target.checked)} className="rounded text-blue-600 focus:ring-blue-500" />
                <span className="text-slate-600">Show Latency Breakdown</span>
              </label>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 w-full max-w-4xl p-6 space-y-8 overflow-y-auto mt-4 pb-32">
        {messages.length === 0 && (
          <div className="text-center text-slate-500 mt-24">
            <h2 className="text-2xl font-semibold mb-3 text-slate-700">How can I help you today?</h2>
            <p>Ask about boiler efficiency, motors, compressed air, furnaces, power factor...</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
            <div 
              className={`max-w-[90%] rounded-xl p-5 ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white shadow-md' 
                  : 'bg-white border border-slate-200 shadow-sm'
              }`}
            >
              {msg.role === 'user' ? (
                <p className="whitespace-pre-wrap text-[15px]">{msg.content}</p>
              ) : (
                msg.error ? (
                  <p className="text-red-600 font-medium">{msg.error}</p>
                ) : (
                  <div className="space-y-5">
                    
                    {/* Return Debug Metadata Display */}
                    {(retrievalMode === "research" || showLatency || showChunks || showScores) && msg.data.debug && (
                      <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                        <h4 className="text-sm font-bold text-slate-700 mb-3 border-b border-slate-200 pb-2">Developer Details</h4>
                        
                        {msg.data.debug.warnings?.length > 0 && (
                          <div className="mb-4">
                            {msg.data.debug.warnings.map((w: string, i: number) => (
                              <p key={i} className="text-xs text-amber-600 bg-amber-50 p-2 rounded border border-amber-100 mb-1">{w}</p>
                            ))}
                          </div>
                        )}

                        {renderRetrievalPath(msg.data.retrieval_plan)}

                        {msg.data.debug.latency_ms && Object.keys(msg.data.debug.latency_ms).length > 0 && (
                          <details className="text-xs text-slate-600 mb-2">
                            <summary className="cursor-pointer font-medium hover:text-slate-800">Latency Breakdown ({msg.data.debug.latency_ms.total?.toFixed(0)}ms)</summary>
                            <pre className="mt-2 p-2 bg-white rounded border border-slate-200 overflow-x-auto">
                              {JSON.stringify(msg.data.debug.latency_ms, null, 2)}
                            </pre>
                          </details>
                        )}

                        {msg.data.debug.scores?.length > 0 && (
                          <details className="text-xs text-slate-600 mb-2">
                            <summary className="cursor-pointer font-medium hover:text-slate-800">Retrieval Scores</summary>
                            <pre className="mt-2 p-2 bg-white rounded border border-slate-200 overflow-x-auto">
                              {JSON.stringify(msg.data.debug.scores, null, 2)}
                            </pre>
                          </details>
                        )}
                        
                        {msg.data.debug.retrieved_chunks?.length > 0 && (
                          <details className="text-xs text-slate-600">
                            <summary className="cursor-pointer font-medium hover:text-slate-800">Retrieved Chunks ({msg.data.debug.retrieved_chunks.length})</summary>
                            <div className="mt-2 space-y-2">
                              {msg.data.debug.retrieved_chunks.map((chunk: any, i: number) => (
                                <div key={i} className="bg-white p-2 rounded border border-slate-200 overflow-x-auto">
                                  <p className="font-mono text-[10px] text-blue-600 mb-1">{chunk.chunk_id}</p>
                                  <p className="text-[11px] whitespace-pre-wrap">{chunk.text}</p>
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    )}

                    {/* Answer Area */}
                    <div className="prose prose-slate prose-sm max-w-none">
                      <ReactMarkdown>{msg.data.answer}</ReactMarkdown>
                    </div>
                    
                    {/* Citations Area */}
                    {msg.data.citations && msg.data.citations.length > 0 && (
                      <div className="pt-4 mt-2">
                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Sources</h4>
                        <div className="flex flex-wrap gap-2">
                          {msg.data.citations.map((cit: any, i: number) => (
                            <details key={i} className="group relative text-xs bg-slate-50 border border-slate-200 rounded-md p-2 max-w-full lg:max-w-[48%] flex-1">
                              <summary className="cursor-pointer font-medium text-blue-600 hover:text-blue-800 transition-colors list-none flex items-center gap-1">
                                <span className="text-slate-400 mr-1">[{i+1}]</span>
                                {cit.book}, {cit.chapter && <span className="truncate max-w-[150px] inline-block align-bottom">{cit.chapter}</span>}
                                {cit.page > 0 && `, pg ${cit.page}`}
                              </summary>
                              <div className="mt-3 text-slate-600 font-mono text-[10px] bg-white p-3 rounded border border-slate-100 shadow-sm whitespace-pre-wrap">
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
          <div className="flex items-start">
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-100"></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce delay-200"></div>
              <span className="text-sm text-slate-500 ml-2 font-medium">Searching BEE Manuals...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="w-full bg-white/80 backdrop-blur-md border-t border-slate-200 p-4 fixed bottom-0 z-20 shadow-[0_-10px_20px_-10px_rgba(0,0,0,0.05)]">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex space-x-3">
          <input
            type="text"
            className="flex-1 bg-slate-50 border border-slate-200 rounded-xl px-5 py-4 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all shadow-inner text-[15px]"
            placeholder="Ask an engineering question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="bg-blue-600 text-white px-8 py-4 rounded-xl font-medium hover:bg-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg active:scale-95"
          >
            Send
          </button>
        </form>
      </div>
    </main>
  );
}
