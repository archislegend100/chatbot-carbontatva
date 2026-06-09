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
    if (plan.use_dense) paths.push("Semantic");
    if (plan.use_sparse) paths.push("Sparse");
    if (plan.use_colbert) paths.push("ColBERT");
    if (plan.use_multi_query) paths.push("Multi-Query");
    if (plan.use_hyde) paths.push("HyDE");
    if (plan.use_reranking) paths.push("Reranked");

    return (
      <div className="flex flex-wrap gap-2 mt-2 mb-4">
        {paths.map(p => (
          <span key={p} className="px-2 py-0.5 bg-indigo-500/10 text-indigo-300 text-xs font-semibold rounded-full border border-indigo-500/20 shadow-[0_0_10px_rgba(99,102,241,0.1)]">
            {p}
          </span>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen relative flex flex-col items-center bg-[#05050f] overflow-hidden">
      
      {/* Dynamic Background Gradients */}
      <div className="fixed top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-indigo-600/20 blur-[120px] pointer-events-none mix-blend-screen animate-pulse duration-1000"></div>
      <div className="fixed bottom-[-20%] right-[-10%] w-[60%] h-[60%] rounded-full bg-blue-600/10 blur-[150px] pointer-events-none mix-blend-screen"></div>

      {/* Header - Glassmorphic */}
      <header className="w-full max-w-5xl py-4 px-6 mt-6 rounded-2xl flex flex-col md:flex-row justify-between items-center z-20 bg-white/[0.03] backdrop-blur-xl border border-white/[0.08] shadow-2xl">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <span className="text-white font-bold text-xl tracking-tighter">CT</span>
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">CarbonTatva</h1>
            <p className="text-xs font-medium text-indigo-400 tracking-widest uppercase">Energy Intelligence</p>
          </div>
        </div>
        
        <div className="flex flex-col items-end gap-3 mt-4 md:mt-0">
          <div className="flex items-center bg-black/40 rounded-lg p-1 border border-white/5 shadow-inner">
            {(["auto", "fast", "deep", "research"] as RetrievalMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setRetrievalMode(mode)}
                className={`px-5 py-2 text-sm font-semibold rounded-md capitalize transition-all duration-300 ${
                  retrievalMode === mode 
                    ? 'bg-gradient-to-r from-indigo-500 to-blue-600 text-white shadow-md' 
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
          <button 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs font-medium text-slate-500 hover:text-indigo-400 transition-colors uppercase tracking-wider flex items-center gap-1"
          >
            {showAdvanced ? "Hide Advanced Settings" : "Show Advanced Settings"}
            <svg className={`w-3 h-3 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
          </button>
        </div>
      </header>

      {/* Advanced Settings Panel */}
      <div className={`w-full max-w-5xl overflow-hidden transition-all duration-500 ease-in-out z-10 ${showAdvanced ? 'max-h-96 opacity-100 mt-4' : 'max-h-0 opacity-0 mt-0'}`}>
        <div className="bg-white/[0.02] backdrop-blur-lg border border-white/[0.05] rounded-2xl p-6 grid grid-cols-1 md:grid-cols-2 gap-8 shadow-2xl">
          <div className="space-y-4">
            <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-widest mb-2">Overrides</h3>
            {[
              { label: "Force ColBERT (Precision)", state: forceColbert, setter: setForceColbert },
              { label: "Force HyDE (Semantic Expansion)", state: forceHyde, setter: setForceHyde },
              { label: "Force Multi-Query", state: forceMultiQuery, setter: setForceMultiQuery },
            ].map((opt, i) => (
              <label key={i} className="flex items-center gap-3 cursor-pointer group">
                <div className={`w-5 h-5 rounded flex items-center justify-center transition-all duration-300 ${opt.state ? 'bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]' : 'bg-slate-800 group-hover:bg-slate-700'}`}>
                  {opt.state && <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>}
                </div>
                <span className="text-sm text-slate-300 group-hover:text-white transition-colors">{opt.label}</span>
              </label>
            ))}
          </div>
          <div className="space-y-4">
            <h3 className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-2">Diagnostics</h3>
            {[
              { label: "Show Retrieved Chunks", state: showChunks, setter: setShowChunks },
              { label: "Show Retrieval Scores", state: showScores, setter: setShowScores },
              { label: "Show Latency Breakdown", state: showLatency, setter: setShowLatency },
            ].map((opt, i) => (
              <label key={i} className="flex items-center gap-3 cursor-pointer group">
                <div className={`w-5 h-5 rounded flex items-center justify-center transition-all duration-300 ${opt.state ? 'bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)]' : 'bg-slate-800 group-hover:bg-slate-700'}`}>
                  {opt.state && <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>}
                </div>
                <span className="text-sm text-slate-300 group-hover:text-white transition-colors">{opt.label}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 w-full max-w-4xl p-4 sm:p-6 space-y-8 overflow-y-auto mt-4 pb-48 z-10 scrollbar-hide">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full mt-10">
            <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-indigo-500/20 to-blue-500/20 flex items-center justify-center mb-8 border border-white/10 shadow-[0_0_50px_rgba(99,102,241,0.15)] animate-pulse duration-1000">
              <svg className="w-10 h-10 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h2 className="text-4xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent mb-4 text-center">How can I help you today?</h2>
            <p className="text-lg text-slate-400 max-w-lg text-center font-medium">
              Ask me anything about boiler efficiency, motor performance, compressed air troubleshooting, or BEE compliance.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role !== 'user' && (
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 mr-4 mt-1 flex-shrink-0">
                <span className="text-white text-xs font-bold">CT</span>
              </div>
            )}
            
            <div 
              className={`max-w-[85%] rounded-2xl p-6 shadow-2xl ${
                msg.role === 'user' 
                  ? 'bg-gradient-to-br from-indigo-600 to-blue-600 text-white rounded-tr-sm' 
                  : 'bg-white/[0.03] backdrop-blur-md border border-white/[0.08] text-slate-200 rounded-tl-sm'
              }`}
            >
              {msg.role === 'user' ? (
                <p className="whitespace-pre-wrap text-[16px] leading-relaxed font-medium">{msg.content}</p>
              ) : (
                msg.error ? (
                  <div className="flex items-center gap-3 text-red-400 bg-red-400/10 p-4 rounded-xl border border-red-400/20">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    <p className="text-sm font-semibold">{msg.error}</p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Return Debug Metadata Display */}
                    {(retrievalMode === "research" || showLatency || showChunks || showScores) && msg.data.debug && (
                      <div className="bg-black/50 p-5 rounded-xl border border-indigo-500/20 mb-6 font-mono relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-indigo-500 to-blue-500"></div>
                        <h4 className="text-xs text-indigo-400 font-bold mb-3 flex items-center gap-2">
                          <svg className="w-4 h-4 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
                          DIAGNOSTICS
                        </h4>
                        
                        {renderRetrievalPath(msg.data.retrieval_plan)}

                        {msg.data.debug.latency_ms && Object.keys(msg.data.debug.latency_ms).length > 0 && (
                          <div className="text-xs text-slate-400 bg-black/40 p-3 rounded-lg border border-white/5 mt-3">
                            <strong className="text-indigo-300">Total Latency: </strong> {msg.data.debug.latency_ms.total?.toFixed(0)}ms
                          </div>
                        )}
                      </div>
                    )}

                    {/* Answer Area */}
                    <div className="prose prose-invert prose-indigo max-w-none prose-p:leading-relaxed prose-p:text-[16px] prose-a:text-indigo-400 prose-pre:bg-black/50 prose-pre:border prose-pre:border-white/10 prose-headings:text-white">
                      <ReactMarkdown>{msg.data.answer}</ReactMarkdown>
                    </div>
                    
                    {/* Citations Area */}
                    {msg.data.citations && msg.data.citations.length > 0 && (
                      <div className="pt-6 mt-6 border-t border-white/10">
                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>
                          Grounding Sources
                        </h4>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {msg.data.citations.map((cit: any, i: number) => (
                            <div key={i} className="group text-sm bg-white/[0.02] hover:bg-white/[0.05] border border-white/[0.05] hover:border-indigo-500/30 transition-all duration-300 rounded-xl p-4 cursor-default">
                              <div className="font-semibold text-indigo-300 flex items-center gap-2 mb-2">
                                <span className="flex-shrink-0 w-5 h-5 rounded-md bg-indigo-500/20 text-indigo-300 flex items-center justify-center text-[10px]">{i+1}</span>
                                <span className="truncate">{cit.book}</span>
                                {cit.page > 0 && <span className="text-slate-400 text-[10px] bg-black/40 px-2 py-0.5 rounded-full ml-auto">Pg {cit.page}</span>}
                              </div>
                              <div className="text-slate-400 text-xs line-clamp-3 leading-relaxed">
                                {cit.text_snippet}
                              </div>
                            </div>
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
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 mr-4 mt-1">
              <span className="text-white text-xs font-bold">CT</span>
            </div>
            <div className="bg-white/[0.03] backdrop-blur-md border border-white/[0.08] rounded-2xl rounded-tl-sm p-5 shadow-xl flex items-center gap-3">
              <div className="flex space-x-1.5">
                <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-150"></div>
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-300"></div>
              </div>
              <span className="text-sm font-medium text-slate-400">Analyzing engineering documents...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Floating Input Area */}
      <div className="w-full max-w-4xl fixed bottom-8 z-30 px-4">
        <form onSubmit={handleSubmit} className="relative flex items-center">
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/20 to-blue-500/20 blur-xl rounded-full"></div>
          <input
            type="text"
            className="relative w-full bg-[#0a0a14]/90 backdrop-blur-xl border border-white/10 focus:border-indigo-500/50 text-white placeholder:text-slate-500 rounded-full pl-6 pr-32 py-5 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 transition-all text-[16px] shadow-2xl"
            placeholder="Ask a technical question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="absolute right-2 top-2 bottom-2 bg-gradient-to-r from-indigo-500 to-blue-600 text-white px-8 rounded-full font-bold hover:shadow-[0_0_20px_rgba(99,102,241,0.4)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none active:scale-[0.98]"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
