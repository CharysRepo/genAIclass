import { auth, signIn, signOut } from "@/auth"
import ChatClient from "./ChatClient"

export default async function ChatPage() {
    const session = await auth()

    if (!session) {
        return (
            <div className="flex h-screen flex-col items-center justify-center bg-[#0d1117] text-white">
                <h1 className="text-2xl font-bold mb-6">Local ChatGPT</h1>
                <form action={async () => { "use server"; await signIn("google") }}>
                    <button className="bg-white text-black px-6 py-2 rounded-md font-medium hover:bg-gray-200 transition">
                        Sign in with Google
                    </button>
                </form>
            </div>
        )
    }

    return (
        <div className="h-screen flex flex-col bg-[#0d1117]">
            {/* Simple Header with Logout */}
            <header className="flex justify-between items-center p-4 border-b border-gray-800 bg-[#161b22]">
                <div className="flex items-center gap-3">
                    <img src={session.user?.image || ''} className="w-8 h-8 rounded-full" alt="avatar" referrerPolicy="no-referrer" />
                    <span className="text-sm font-medium text-gray-300">{session.user?.name}</span>
                </div>
                <form action={async () => { "use server"; await signOut() }}>
                    <button className="text-xs text-gray-500 hover:text-red-400">Sign Out</button>
                </form>
            </header>

            {/* The actual Chat logic (Client Component) */}
            <ChatClient userId={session.user?.email || "anonymous"} />
        </div>
    )
}
