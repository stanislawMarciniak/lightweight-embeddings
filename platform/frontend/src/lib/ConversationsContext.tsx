import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode
} from "react";
import { Conversation, apiDeleteConversation, apiGetConversations } from "./api";
import { supabase } from "./supabaseClient";

interface ConversationsContextValue {
  conversations: Conversation[];
  loading: boolean;
  reload: () => Promise<void>;
  remove: (id: string) => Promise<void>;
}

const ConversationsContext = createContext<ConversationsContextValue | null>(null);

/**
 * Shared store for the user's chat conversations so the Sidebar (which lists
 * them) and the ChatPage (which creates/opens them) stay in sync without
 * prop-drilling. Refetches when the auth session changes.
 */
export function ConversationsProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(false);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiGetConversations();
      setConversations(data);
    } catch {
      // Not authenticated yet / transient failure: keep the list empty.
      setConversations([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const remove = useCallback(async (id: string) => {
    await apiDeleteConversation(id);
    setConversations((prev) => prev.filter((c) => c.id !== id));
  }, []);

  useEffect(() => {
    reload();
    const {
      data: { subscription }
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session) {
        reload();
      } else {
        setConversations([]);
      }
    });
    return () => subscription.unsubscribe();
  }, [reload]);

  return (
    <ConversationsContext.Provider value={{ conversations, loading, reload, remove }}>
      {children}
    </ConversationsContext.Provider>
  );
}

export function useConversations(): ConversationsContextValue {
  const ctx = useContext(ConversationsContext);
  if (!ctx) {
    throw new Error("useConversations must be used within a ConversationsProvider");
  }
  return ctx;
}
