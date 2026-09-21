import NextAuth from "next-auth"
import Google from "next-auth/providers/google"

export const { handlers, auth, signIn, signOut } = NextAuth({
    providers: [Google],
    callbacks: {
        authorized({ auth, request: { nextUrl } }) {
            const isLoggedIn = !!auth?.user;
            const isOnChat = nextUrl.pathname.startsWith('/');
            if (isOnChat) {
                if (isLoggedIn) return true;
                return false; // Redirect unauthenticated users to login
            }
            return true;
        },
    },
})
