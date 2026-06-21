import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { supabase } from "./lib/supabaseClient";
import { ConversationsProvider } from "./lib/ConversationsContext";
import { LoginPage } from "./pages/LoginPage";
import { ChatPage } from "./pages/ChatPage";
import { FAQPage } from "./pages/FAQPage";
import { DocumentsPage } from "./pages/DocumentsPage";

function RequireAuth({ children }: { children: JSX.Element }) {
  const [loading, setLoading] = useState(true);
  const [authenticated, setAuthenticated] = useState(false);
  const location = useLocation();

  useEffect(() => {
    let isMounted = true;

    supabase.auth.getSession().then(({ data }) => {
      if (!isMounted) return;
      setAuthenticated(!!data.session);
      setLoading(false);
    });

    const {
      data: { subscription }
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!isMounted) return;
      setAuthenticated(!!session);
    });

    return () => {
      isMounted = false;
      subscription.unsubscribe();
    };
  }, []);

  if (loading) {
    return <div className="auth-layout muted">Loading...</div>;
  }

  if (!authenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
}

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();

  // Only redirect when on login: send to chat (or saved "from") when logged in, else stay on login
  useEffect(() => {
    if (location.pathname !== "/login") return;

    supabase.auth.getSession().then(({ data }) => {
      if (data.session) {
        const from = (location.state as { from?: { pathname?: string } })?.from?.pathname;
        navigate(from && from !== "/login" ? from : "/chat", { replace: true });
      }
    });
  }, [navigate, location.pathname, location.state]);

  return (
    <ConversationsProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/chat"
          element={
            <RequireAuth>
              <ChatPage />
            </RequireAuth>
          }
        />
        <Route
          path="/chat/:conversationId"
          element={
            <RequireAuth>
              <ChatPage />
            </RequireAuth>
          }
        />
        <Route
          path="/faq"
          element={
            <RequireAuth>
              <FAQPage />
            </RequireAuth>
          }
        />
        <Route
          path="/documents"
          element={
            <RequireAuth>
              <DocumentsPage />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </ConversationsProvider>
  );
}

