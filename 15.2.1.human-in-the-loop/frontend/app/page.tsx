'use client';
import React, { useState } from 'react';

interface ToolRequest {
  tool_call_id: string;
  tool_name: string;
  args: any;
}

type Approvals = Record<string, boolean>;

export default function ChatApp() {
  const [userInput, setUserInput] = useState("");
  const [pendingApprovals, setPendingApprovals] = useState<ToolRequest[]>([]);
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const callAgent = async (msg: string, approvals: Approvals = {}) => {
    setIsLoading(true);
    try {
      const res = await fetch('http://localhost:8000/chat/greet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, approvals: approvals }),
      });
      const data = await res.json();
      if (data.status === "requires_approval") {
        setPendingApprovals(data.approvals || []);
        setResponse("");
      } else {
        setResponse(data.output);
        setPendingApprovals([]);
      }
    } catch (err) {
      console.error("Fetch error:", err);
      setResponse("⚠️ Error connecting to backend. Please ensure the server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{
      maxWidth: '650px',
      margin: '40px auto',
      padding: '32px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      backgroundColor: '#ffffff',
      borderRadius: '16px',
      boxShadow: '0 4px 24px rgba(0, 0, 0, 0.06)',
      border: '1px solid #f0f0f0'
    }}>
      {/* Header */}
      <div style={{ marginBottom: '28px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: 700, color: '#1a1a1a', margin: '0 0 6px 0' }}>
          PydanticAI Human-In-The-Loop
        </h1>
        <p style={{ fontSize: '14px', color: '#666', margin: 0 }}>
          Agent authorization dashboard
        </p>
      </div>

      {/* Input Group */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
        <input
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Ask the agent..."
          disabled={isLoading}
          style={{
            flex: 1,
            padding: '12px 16px',
            borderRadius: '10px',
            border: '1px solid #e0e0e0',
            fontSize: '15px',
            backgroundColor: isLoading ? '#f8f9fa' : '#ffffff',
            outline: 'none',
            transition: 'border-color 0.2s'
          }}
        />
        <button
          onClick={() => callAgent(userInput)}
          disabled={isLoading || !userInput.trim()}
          style={{
            backgroundColor: isLoading || !userInput.trim() ? '#a5c4ff' : '#2563eb',
            color: 'white',
            padding: '12px 24px',
            borderRadius: '10px',
            border: 'none',
            fontSize: '15px',
            fontWeight: 600,
            cursor: isLoading || !userInput.trim() ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.2s',
          }}
        >
          {isLoading ? "Thinking..." : "Send"}
        </button>
      </div>

      {/* Approval Section */}
      {pendingApprovals.length > 0 && (
        <div style={{
          backgroundColor: '#fffbeb',
          border: '1px solid #fef3c7',
          padding: '20px',
          borderRadius: '12px',
          marginBottom: '24px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <span style={{ fontSize: '18px' }}>🛡️</span>
            <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#92400e', margin: 0 }}>
              Authorization Required
            </h3>
          </div>

          {pendingApprovals.map(req => (
            <div key={req.tool_call_id} style={{
              backgroundColor: '#ffffff',
              border: '1px solid #f59e0b20',
              borderRadius: '8px',
              padding: '16px',
              boxShadow: '0 2px 8px rgba(245, 158, 11, 0.04)'
            }}>
              <div style={{ marginBottom: '12px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: '#b45309', backgroundColor: '#fef3c7', padding: '4px 8px', borderRadius: '4px' }}>
                  {req.tool_name}
                </span>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '12px', fontWeight: 600, color: '#666', marginBottom: '4px' }}>Arguments</div>
                <pre style={{
                  margin: 0,
                  padding: '10px',
                  backgroundColor: '#f8f9fa',
                  borderRadius: '6px',
                  fontSize: '13px',
                  color: '#333',
                  fontFamily: 'monospace',
                  overflowX: 'auto'
                }}>
                  {JSON.stringify(req.args, null, 2)}
                </pre>
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={() => callAgent(userInput, { [req.tool_call_id]: false })}
                  style={{
                    flex: 1,
                    backgroundColor: '#ffffff',
                    color: '#dc2626',
                    padding: '10px 16px',
                    borderRadius: '8px',
                    border: '1px solid #fee2e2',
                    fontWeight: 600,
                    fontSize: '14px',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                >
                  Reject
                </button>
                <button
                  onClick={() => callAgent(userInput, { [req.tool_call_id]: true })}
                  style={{
                    flex: 2,
                    backgroundColor: '#059669',
                    color: 'white',
                    padding: '10px 16px',
                    borderRadius: '8px',
                    border: 'none',
                    fontWeight: 600,
                    fontSize: '14px',
                    cursor: 'pointer',
                    transition: 'background-color 0.2s'
                  }}
                >
                  Approve & Execute
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Final Response */}
      {response && (
        <div style={{
          padding: '20px',
          backgroundColor: '#f8f9fa',
          borderRadius: '12px',
          border: '1px solid #e9ecef'
        }}>
          <div style={{
            fontSize: '12px',
            fontWeight: 700,
            color: '#6c757d',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginBottom: '8px'
          }}>
            Agent Output
          </div>
          <p style={{ margin: 0, fontSize: '15px', color: '#212529', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
            {response}
          </p>
        </div>
      )}
    </div>
  );
}
