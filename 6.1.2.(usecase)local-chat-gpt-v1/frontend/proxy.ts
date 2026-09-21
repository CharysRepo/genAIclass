import { auth } from "@/auth"

export default auth((req) => {
    const isLoggedIn = !!req.auth;
    const isOnChat = req.nextUrl.pathname.startsWith('/');

    // If the user tries to access /chat but isn't logged in, 
    // redirect them to the Google sign-in page automatically.
    if (isOnChat && !isLoggedIn) {
        return Response.redirect(new URL('/api/auth/signin', req.nextUrl));
    }
});

// Configure which paths the middleware should run on
export const config = {
    matcher: ['/chat/:path*'],
};
