"use client";

import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Settings, Plus, MessageSquare, ChevronDown, Cpu, ChevronRight, Activity, Database, CheckCircle2, AlertCircle } from 'lucide-react';

type RetrievalMode = "auto" | "fast" | "deep" | "research";

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
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

  // Expanded thought process state per message
  const [expandedThoughts, setExpandedThoughts] = useState<Record<number, boolean>>({});

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

  const handleNewChat = () => {
    setMessages([]);
    setExpandedThoughts({});
  };

  const toggleThought = (idx: number) => {
    setExpandedThoughts(prev => ({...prev, [idx]: !prev[idx]}));
  };

  const renderAgenticTools = (plan: any, latency: any, idx: number) => {
    if (!plan && !latency) return null;
    const isExpanded = expandedThoughts[idx];

    return (
      <div className="mb-4 mt-2">
        <button 
          onClick={() => toggleThought(idx)}
          className="flex items-center gap-2 text-xs font-mono text-indigo-400/80 hover:text-indigo-300 transition-colors bg-white/5 hover:bg-white/10 px-3 py-1.5 rounded-full border border-white/5"
        >
          {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          <Cpu className="w-3 h-3" />
          <span>Agentic Execution Paths</span>
        </button>

        {isExpanded && (
          <div className="mt-3 p-4 rounded-xl bg-black/40 border border-white/5 shadow-inner">
            <h4 className="text-[10px] uppercase tracking-widest text-slate-500 mb-3 font-semibold">Active Tools & Pipelines</h4>
            <div className="grid grid-cols-2 gap-2">
              {plan?.use_dense && (
                <div className="flex items-center gap-2 text-xs text-slate-300 bg-white/5 p-2 rounded border border-white/5">
                  <Database className="w-3 h-3 text-blue-400" /> Dense Semantic Search
                </div>
              )}
              {plan?.use_sparse && (
                <div className="flex items-center gap-2 text-xs text-slate-300 bg-white/5 p-2 rounded border border-white/5">
                  <Activity className="w-3 h-3 text-green-400" /> Sparse Keyword Match
                </div>
              )}
              {plan?.use_colbert && (
                <div className="flex items-center gap-2 text-xs text-slate-300 bg-white/5 p-2 rounded border border-white/5">
                  <CheckCircle2 className="w-3 h-3 text-purple-400" /> ColBERT Token Scoring
                </div>
              )}
              {plan?.use_multi_query && (
                <div className="flex items-center gap-2 text-xs text-slate-300 bg-white/5 p-2 rounded border border-white/5">
                  <MessageSquare className="w-3 h-3 text-indigo-400" /> Query Decomposition
                </div>
              )}
              {plan?.use_hyde && (
                <div className="flex items-center gap-2 text-xs text-slate-300 bg-white/5 p-2 rounded border border-white/5">
                  <AlertCircle className="w-3 h-3 text-orange-400" /> HyDE Expansion
                </div>
              )}
              {plan?.use_reranking && (
                <div className="flex items-center gap-2 text-xs text-slate-300 bg-white/5 p-2 rounded border border-white/5">
                  <Activity className="w-3 h-3 text-rose-400" /> Cross-Encoder Reranking
                </div>
              )}
            </div>

            {latency && Object.keys(latency).length > 0 && (
              <div className="mt-4 pt-3 border-t border-white/5 flex justify-between items-center text-[11px] font-mono text-slate-400">
                <span>Total Pipeline Latency</span>
                <span className="text-indigo-300">{latency.total?.toFixed(0)}ms</span>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-[#05050f] overflow-hidden text-slate-200">
      
      {/* Background Orbs */}
      <div className="fixed top-[-20%] left-[10%] w-[40%] h-[40%] rounded-full bg-indigo-600/10 blur-[120px] pointer-events-none mix-blend-screen animate-pulse duration-1000"></div>
      <div className="fixed bottom-[-10%] right-[10%] w-[50%] h-[50%] rounded-full bg-blue-600/5 blur-[150px] pointer-events-none mix-blend-screen"></div>

      {/* Sidebar */}
      <div className={`flex flex-col bg-[#0a0a14] border-r border-white/5 transition-all duration-300 z-20 ${isSidebarOpen ? 'w-64' : 'w-0 opacity-0'}`}>
        <div className="p-4 flex-1 flex flex-col gap-4 overflow-y-auto min-w-[256px]">
          <button 
            onClick={handleNewChat}
            className="flex items-center gap-2 w-full px-4 py-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors border border-white/5 text-sm font-medium"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>

          {/* History Mockup */}
          <div className="flex-1 mt-6">
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3 px-2">History</h3>
            <div className="space-y-1">
              <button className="flex items-center gap-3 w-full px-3 py-2 text-sm text-slate-300 bg-white/5 rounded-lg border border-white/5">
                <MessageSquare className="w-4 h-4 text-indigo-400" />
                <span className="truncate">Current Session</span>
              </button>
            </div>
          </div>

          {/* Settings Section in Sidebar */}
          <div className="pt-4 border-t border-white/5">
            <button 
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 w-full px-3 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors rounded-lg"
            >
              <Settings className="w-4 h-4" />
              Advanced Settings
              <ChevronDown className={`w-4 h-4 ml-auto transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
            </button>
            
            {showAdvanced && (
              <div className="mt-2 p-3 bg-black/40 rounded-xl border border-white/5 space-y-3">
                <h4 className="text-[10px] text-indigo-400 uppercase tracking-wider font-semibold">Overrides</h4>
                {[
                  { label: "Force ColBERT", state: forceColbert, setter: setForceColbert },
                  { label: "Force HyDE", state: forceHyde, setter: setForceHyde },
                  { label: "Force Multi-Query", state: forceMultiQuery, setter: setForceMultiQuery },
                ].map((opt, i) => (
                  <label key={i} className="flex items-center gap-2 cursor-pointer group">
                    <input type="checkbox" className="hidden" checked={opt.state} onChange={() => opt.setter(!opt.state)} />
                    <div className={`w-4 h-4 rounded-sm flex items-center justify-center border transition-all ${opt.state ? 'bg-indigo-500 border-indigo-500' : 'border-white/20 group-hover:border-white/40'}`}>
                      {opt.state && <CheckCircle2 className="w-3 h-3 text-white" />}
                    </div>
                    <span className="text-xs text-slate-300">{opt.label}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Area */}
      <div className="flex-1 flex flex-col relative z-10 w-full">
        {/* Top Header */}
        <header className="h-14 flex items-center px-4 justify-between border-b border-white/5 bg-transparent backdrop-blur-sm sticky top-0 z-20">
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-2 hover:bg-white/5 rounded-lg transition-colors text-slate-400 hover:text-slate-200"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" /></svg>
            </button>
            <h1 className="text-sm font-semibold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">CarbonTatva Energy Intelligence</h1>
          </div>
        </header>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 pb-40 scrollbar-hide">
          <div className="max-w-3xl mx-auto space-y-8">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full mt-24">
                <div className="w-20 h-20 rounded-3xl bg-gradient-to-tr from-indigo-500/20 to-blue-500/20 flex items-center justify-center mb-6 border border-white/10 shadow-[0_0_50px_rgba(99,102,241,0.15)] animate-pulse duration-1000">
                  <span className="text-white font-bold text-2xl tracking-tighter">CT</span>
                </div>
                <h2 className="text-3xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent mb-3 text-center">How can I help you today?</h2>
                <p className="text-md text-slate-500 max-w-md text-center font-medium">
                  Ask me anything about boiler efficiency, motor performance, or BEE compliance.
                </p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role !== 'user' && (
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 mr-4 mt-1 flex-shrink-0">
                    <span className="text-white text-[10px] font-bold">CT</span>
                  </div>
                )}
                
                <div className={`max-w-[85%] ${msg.role === 'user' ? 'bg-white/10' : 'bg-transparent'} rounded-2xl ${msg.role === 'user' ? 'px-5 py-3' : 'py-2'} text-slate-200`}>
                  {msg.role === 'user' ? (
                    <p className="whitespace-pre-wrap text-[15px] font-medium leading-relaxed">{msg.content}</p>
                  ) : (
                    msg.error ? (
                      <div className="flex items-center gap-3 text-red-400 bg-red-400/10 p-4 rounded-xl border border-red-400/20">
                        <AlertCircle className="w-5 h-5" />
                        <p className="text-sm font-semibold">{msg.error}</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {/* Agentic Tools Display */}
                        {msg.data.debug && renderAgenticTools(msg.data.retrieval_plan, msg.data.debug.latency_ms, idx)}

                        {/* Answer Area */}
                        <div className="prose prose-invert prose-indigo max-w-none prose-p:leading-relaxed prose-p:text-[15px] prose-a:text-indigo-400 prose-pre:bg-black/50 prose-pre:border prose-pre:border-white/10 prose-headings:text-white">
                          <ReactMarkdown>{msg.data.answer}</ReactMarkdown>
                        </div>
                        
                        {/* Citations Area */}
                        {msg.data.citations && msg.data.citations.length > 0 && (
                          <div className="pt-4 mt-6 flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
                            {msg.data.citations.map((cit: any, i: number) => (
                              <div key={i} className="flex-shrink-0 w-64 group text-sm bg-white/[0.03] border border-white/[0.05] hover:border-indigo-500/30 transition-all duration-300 rounded-xl p-3 cursor-default">
                                <div className="font-semibold text-indigo-300 flex items-center gap-2 mb-1">
                                  <span className="truncate text-xs">{cit.book}</span>
                                  {cit.page > 0 && <span className="text-slate-400 text-[9px] bg-black/40 px-1.5 py-0.5 rounded-full ml-auto">Pg {cit.page}</span>}
                                </div>
                                <div className="text-slate-400 text-[11px] line-clamp-2 leading-relaxed">
                                  {cit.text_snippet}
                                </div>
                              </div>
                            ))}
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
                  <span className="text-white text-[10px] font-bold">CT</span>
                </div>
                <div className="py-2 flex items-center gap-3">
                  <div className="flex space-x-1">
                    <div className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce"></div>
                    <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-150"></div>
                    <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce delay-300"></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Floating Input Area (ChatGPT Style) */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-[#05050f] via-[#05050f] to-transparent pt-10">
          <div className="max-w-3xl mx-auto relative">
            <form onSubmit={handleSubmit} className="relative flex items-end gap-2 bg-[#0a0a14]/90 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl p-2 focus-within:border-indigo-500/50 transition-colors">
              
              {/* Inline Dropdown for Retrieval Mode */}
              <div className="relative flex-shrink-0 group">
                <select 
                  value={retrievalMode}
                  onChange={(e) => setRetrievalMode(e.target.value as RetrievalMode)}
                  className="appearance-none bg-white/5 hover:bg-white/10 text-xs font-semibold text-indigo-300 px-3 py-2 pl-3 pr-8 rounded-lg cursor-pointer outline-none border border-transparent transition-colors h-10"
                >
                  <option value="auto" className="bg-slate-900 text-white">Auto</option>
                  <option value="fast" className="bg-slate-900 text-white">Fast</option>
                  <option value="deep" className="bg-slate-900 text-white">Deep</option>
                  <option value="research" className="bg-slate-900 text-white">Research</option>
                </select>
                <ChevronDown className="absolute right-2 top-3 w-4 h-4 text-indigo-400 pointer-events-none group-hover:text-indigo-300" />
              </div>

              <textarea
                className="flex-1 bg-transparent text-white placeholder:text-slate-500 resize-none py-2 px-2 focus:outline-none min-h-[44px] max-h-32 text-[15px] scrollbar-hide"
                placeholder="Message CarbonTatva..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                disabled={loading}
                rows={1}
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center hover:bg-indigo-500 transition-colors disabled:opacity-50 disabled:bg-white/10 disabled:text-white/30 flex-shrink-0"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" /></svg>
              </button>
            </form>
            <div className="text-center mt-2">
              <span className="text-[10px] text-slate-500 font-medium">CarbonTatva can make mistakes. Verify important engineering decisions.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
