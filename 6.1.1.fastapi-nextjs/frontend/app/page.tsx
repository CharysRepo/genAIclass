'use client';
import { useState } from 'react';

export default function Home() {
  const [name, setName] = useState('');
  const [message, setMessage] = useState('');

  const handleGreet = async () => {
    if (!name) return;
    try {
      // Ensure the URL is correct for your backend server
      const response = await fetch('http://localhost:8000/api/greet', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: name }),
      });
      const data = await response.json();
      setMessage(data.message);
    } catch (err) {
      console.error("Fetch error:", err);
      setMessage("Failed to fetch. Check console and server status.");
    }
  };

  return (
    // Main container styling: full screen, centered content, dark background
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      {/* Card container styling: p-8, rounded corners, shadow, light background for contrast */}
      <div className="bg-gray-800 p-8 rounded-lg shadow-xl w-full max-w-md">
        <h1 className="text-3xl font-bold mb-6 text-center text-blue-400">
          Simple Greeting App
        </h1>

        <div className="flex flex-col gap-4">
          <input
            type="text"
            placeholder="Enter your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            // Input styling: full width, padding, border, rounded, dark background, focus ring
            className="w-full p-3 border border-gray-600 rounded-md bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />

          <button
            onClick={handleGreet}
            // Button styling: full width, padding, blue background, hover effect, rounded, shadow
            className="w-full p-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md shadow-md transition duration-200 disabled:opacity-50"
            disabled={!name} // Disable button if name is empty
          >
            Greet
          </button>
        </div>

        {message && (
          // Message display styling: margin top, center align, blue text, padding, light background
          <div className="mt-6 p-4 bg-gray-700 rounded-md text-center">
            <h2 className="text-xl font-medium text-blue-400">{message}</h2>
          </div>
        )}
      </div>
    </div>
  );
}
