'use client';
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, User, Bot, Loader2, History } from 'lucide-react';

export default function ChatClient({ userId }: { userId: string }) {
    const [messages, setMessages] = useState<{ role: 'user' | 'assistant', content: string }[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isInitializing, setIsInitializing] = useState(true); // New state for initial load
    const scrollRef = useRef<HTMLDivElement>(null);
    const API_BASE = process.env.NEXT_PUBLIC_APP_BASE;

    // 1. Load History on Startup
    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const response = await fetch(`${API_BASE}/chat/history/${userId}`);
                if (!response.ok) throw new Error('Failed to fetch history');
                const data = await response.json();
                // Assumes backend returns { history: [{role: '...', content: '...'}, ...] }
                setMessages(data.history || []);
            } catch (error) {
                console.error("Error loading history:", error);
            } finally {
                setIsInitializing(false);
            }
        };

        fetchHistory();
    }, [userId]);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    const sendMessage = async () => {
        if (!input.trim() || isLoading) return;
        const userMessage = input;
        setInput('');
        setIsLoading(true);

        // Add user message to UI immediately
        setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);

        try {
            // Call the POST endpoint
            const response = await fetch(`${API_BASE}/chat/${userId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: userMessage }),
            });

            if (!response.ok) throw new Error('Failed to fetch response');
            const data = await response.json();

            // Add the assistant's full response to UI
            setMessages((prev) => [...prev, { role: 'assistant', content: data.content }]);
        } catch (error) {
            console.error("Error:", error);
            setMessages((prev) => [...prev, { role: 'assistant', content: "Sorry, I encountered an error." }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-screen bg-[#0d1117] text-gray-100 font-sans">
            {/* Header */}
            <header className="p-4 border-b border-gray-800 flex justify-between items-center bg-[#161b22]">
                <h1 className="text-lg font-semibold tracking-tight">Local ChatGPT</h1>
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`} />
                    <span className="text-xs text-gray-400">
                        {isLoading ? 'Processing...' : isInitializing ? 'Loading History...' : 'Ready'}
                    </span>
                </div>
            </header>

            {/* Message Area */}
            <main className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
                {isInitializing && (
                    <div className="flex justify-center items-center h-full">
                        <Loader2 className="animate-spin text-gray-500" size={32} />
                    </div>
                )}

                {!isInitializing && messages.map((m, i) => (
                    <div key={i} className={`flex gap-4 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${m.role === 'user' ? 'bg-blue-600' : 'bg-gray-700'}`}>
                            {m.role === 'user' ? <User size={14} /> : <Bot size={14} />}
                        </div>
                        <div className={`max-w-[80%] p-4 rounded-2xl shadow-sm ${m.role === 'user' ? 'bg-blue-700 text-white rounded-tr-none' : 'bg-[#21262d] text-gray-200 rounded-tl-none border border-gray-700'}`}>
                            <article className="prose prose-invert prose-sm max-w-none">
                                <ReactMarkdown>{m.content}</ReactMarkdown>
                            </article>
                        </div>
                    </div>
                ))}

                {/* Loading Indicator */}
                {isLoading && (
                    <div className="flex gap-4">
                        <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-gray-700">
                            <Bot size={18} />
                        </div>
                        <div className="bg-[#21262d] p-4 rounded-2xl rounded-tl-none border border-gray-700">
                            <Loader2 className="animate-spin text-gray-500" size={20} />
                        </div>
                    </div>
                )}
                <div ref={scrollRef} />
            </main>

            {/* Input Area */}
            <footer className="p-4 bg-[#161b22] border-t border-gray-800">
                <form onSubmit={(e) => { e.preventDefault(); sendMessage(); }} className="max-w-4xl mx-auto flex gap-3 bg-[#0d1117] border border-gray-700 rounded-xl p-2 focus-within:border-blue-500 transition-colors">
                    <input
                        className="flex-1 bg-transparent px-3 py-2 outline-none text-sm disabled:cursor-not-allowed"
                        placeholder={isLoading ? "Waiting for AI..." : "Ask anything..."}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        disabled={isLoading || isInitializing}
                    />
                    <button type="submit" disabled={isLoading || !input.trim() || isInitializing} className="p-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 rounded-lg transition-all">
                        <Send size={18} />
                    </button>
                </form>
            </footer>
        </div>
    );
}
