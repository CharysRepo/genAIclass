'use client';

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, User, Bot, Loader2 } from 'lucide-react';

export default function ChatClient({ userId }: { userId: string }) {
    const [messages, setMessages] = useState<{ role: 'user' | 'assistant', content: string }[]>([]);
    const [input, setInput] = useState('');
    const [isConnected, setIsConnected] = useState(false);
    const [isLoadingHistory, setIsLoadingHistory] = useState(true);
    const [isStreaming, setIsStreaming] = useState(false); // New state to block input
    const ws = useRef<WebSocket | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const API_BASE = process.env.NEXT_PUBLIC_APP_BASE;

    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    useEffect(() => {
        let isMounted = true;
        const loadHistory = async () => {
            try {
                const response = await fetch(`${API_BASE}/chat/history/${userId}`);
                if (response.ok) {
                    const data = await response.json();
                    if (isMounted) {
                        setMessages(Array.isArray(data) ? data : data.history || []);
                    }
                }
            } catch (error) {
                console.error("Failed to load history:", error);
            } finally {
                if (isMounted) setIsLoadingHistory(false);
            }
        };
        loadHistory();
        return () => { isMounted = false; };
    }, [userId]);

    useEffect(() => {
        if (isLoadingHistory) return;

        const socket = new WebSocket(`${API_BASE}/chat/ws/${userId}`);
        ws.current = socket;

        socket.onopen = () => setIsConnected(true);
        socket.onclose = () => {
            setIsConnected(false);
            setIsStreaming(false); // Stop streaming state if connection drops
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'start') {
                setIsStreaming(true); // Block input when assistant starts
                setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);
            } else if (data.type === 'token') {
                setMessages((prev) => {
                    const last = prev[prev.length - 1];
                    if (last?.role === 'assistant') {
                        const updated = { ...last, content: last.content + data.content };
                        return [...prev.slice(0, -1), updated];
                    }
                    return prev;
                });
            } else if (data.type === 'end') {
                setIsStreaming(false); // Unblock input when assistant finishes
            }
        };

        return () => {
            if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
                socket.close();
            }
        };
    }, [userId, isLoadingHistory]);

    const sendMessage = () => {
        if (!input.trim() || !ws.current || isStreaming) return;

        setIsStreaming(true); // Block immediately on send
        setMessages((prev) => [...prev, { role: 'user', content: input }]);
        ws.current.send(JSON.stringify({ content: input }));
        setInput('');
    };

    // Determine if input should be disabled
    const isInputDisabled = !isConnected || isLoadingHistory || isStreaming;

    return (
        <div className="flex flex-col h-screen bg-[#0d1117] text-gray-100 font-sans">
            <header className="p-4 border-b border-gray-800 flex justify-between items-center bg-[#161b22]">
                <h1 className="text-lg font-semibold tracking-tight">Local ChatGPT</h1>
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="text-xs text-gray-400">{isConnected ? 'Connected' : 'Offline'}</span>
                </div>
            </header>

            <main className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
                {messages.map((m, i) => (
                    <div key={i} className={`flex gap-4 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${m.role === 'user' ? 'bg-blue-600' : 'bg-gray-700'}`}>
                            {m.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                        </div>
                        <div className={`max-w-[80%] p-4 rounded-2xl shadow-sm ${m.role === 'user' ? 'bg-blue-700 text-white rounded-tr-none' : 'bg-[#21262d] text-gray-200 rounded-tl-none border border-gray-700'}`}>
                            <article className="prose prose-invert prose-sm max-w-none">
                                <ReactMarkdown>{m.content}</ReactMarkdown>
                                {!m.content && m.role === 'assistant' && <Loader2 className="animate-spin text-gray-500" size={16} />}
                            </article>
                        </div>
                    </div>
                ))}
                <div ref={scrollRef} />
            </main>

            <footer className="p-4 bg-[#161b22] border-t border-gray-800">
                <div className={`max-w-4xl mx-auto flex gap-3 bg-[#0d1117] border border-gray-700 rounded-xl p-2 transition-colors ${isStreaming ? 'opacity-50' : 'focus-within:border-blue-500'}`}>
                    <input
                        className="flex-1 bg-transparent px-3 py-2 outline-none text-sm disabled:cursor-not-allowed"
                        placeholder={isLoadingHistory ? "Loading history..." : isStreaming ? "Assistant is typing..." : "Ask anything..."}
                        value={input}
                        disabled={isInputDisabled}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                    />
                    <button
                        onClick={sendMessage}
                        disabled={isInputDisabled || !input.trim()}
                        className="p-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 rounded-lg transition-all"
                    >
                        <Send size={18} />
                    </button>
                </div>
            </footer>
        </div>
    );
}
