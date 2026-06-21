import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Chat, ChatMessage } from "../components/Chat";
import { Sidebar } from "../components/Sidebar";
import { useConversations } from "../lib/ConversationsContext";
import { apiGetConversationMessages, apiSendChat } from "../lib/api";
import { responseSourceLabel } from "../lib/sourceLabel";

function toMessages(
  items: {
    id: string;
    message: string;
    response: string;
    response_source?: string | null;
    response_document_name?: string | null;
  }[]
): ChatMessage[] {
  const mapped: ChatMessage[] = [];
  for (const c of items) {
    mapped.push({ id: `${c.id}-user`, role: "user", content: c.message });
    mapped.push({
      id: `${c.id}-assistant`,
      role: "assistant",
      content: c.response,
      sourceLabel: responseSourceLabel(c.response_source, c.response_document_name)
    });
  }
  return mapped;
}

export function ChatPage() {
  const { conversationId } = useParams<{ conversationId?: string }>();
  const navigate = useNavigate();
  const { reload } = useConversations();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Load the messages for whichever conversation the URL points at. No id => a
  // fresh, empty chat (a new conversation is created on the first send).
  useEffect(() => {
    let cancelled = false;
    setInput("");
    (async () => {
      if (!conversationId) {
        setMessages([]);
        return;
      }
      try {
        const items = await apiGetConversationMessages(conversationId);
        if (!cancelled) setMessages(toMessages(items));
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("Failed to load messages", err);
        if (!cancelled) setMessages([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  async function handleSend() {
    if (!input.trim() || loading) return;
    const text = input;
    const userMessage: ChatMessage = {
      id: `local-${Date.now()}-user`,
      role: "user",
      content: text
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    try {
      const res = await apiSendChat(text, conversationId ?? null);
      setMessages((prev) => [
        ...prev,
        {
          id: `local-${Date.now()}-assistant`,
          role: "assistant",
          content: res.response,
          sourceLabel: responseSourceLabel(res.response_source, res.response_document_name)
        }
      ]);
      // Keep the sidebar list (titles / ordering) in sync.
      await reload();
      // A new conversation was created server-side: reflect it in the URL.
      if (!conversationId && res.conversation_id) {
        navigate(`/chat/${res.conversation_id}`, { replace: true });
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: `local-${Date.now()}-error`,
          role: "assistant",
          content: err?.message ?? "Failed to send message"
        }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-content">
        <div className="page-header">
          <div className="page-title">Chat</div>
        </div>
        <Chat
          messages={messages}
          input={input}
          setInput={setInput}
          onSend={handleSend}
          sending={loading}
        />
      </main>
    </div>
  );
}
