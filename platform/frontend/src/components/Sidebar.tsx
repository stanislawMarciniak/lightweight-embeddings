import { NavLink, useNavigate, useParams } from "react-router-dom";
import { supabase } from "../lib/supabaseClient";
import { useEffect, useState } from "react";
import { useConversations } from "../lib/ConversationsContext";
import { usePagination } from "../lib/usePagination";
import { Pagination } from "./Pagination";

function linkClassName({ isActive }: { isActive: boolean }) {
  return `sidebar-link ${isActive ? "sidebar-link-active" : ""}`;
}

export function Sidebar() {
  const [email, setEmail] = useState<string | null>(null);
  const navigate = useNavigate();
  const { conversationId } = useParams<{ conversationId?: string }>();
  const { conversations, remove } = useConversations();
  const { pageItems, page, pageCount, rangeStart, rangeEnd, total, next, prev } = usePagination(
    conversations,
    10
  );

  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      setEmail(data.user?.email ?? null);
    });
  }, []);

  async function handleLogout() {
    await supabase.auth.signOut();
    navigate("/login");
  }

  async function handleDeleteChat(id: string) {
    try {
      await remove(id);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Failed to delete conversation", err);
      return;
    }
    if (id === conversationId) {
      navigate("/chat");
    }
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">Semantic Chat</div>

      <nav className="sidebar-nav">
        <NavLink to="/chat" end className={linkClassName}>
          Chat
        </NavLink>
        <NavLink to="/faq" className={linkClassName}>
          FAQ
        </NavLink>
        <NavLink to="/documents" className={linkClassName}>
          Documents
        </NavLink>
      </nav>

      <div className="sidebar-section">
        <div className="sidebar-section-header">
          <span className="sidebar-section-title">Chats</span>
          <NavLink to="/chat" end className="sidebar-newchat" title="Start a new chat">
            + New
          </NavLink>
        </div>

        <div className="sidebar-chats">
          {total === 0 && <div className="muted small sidebar-empty">No chats yet.</div>}
          {pageItems.map((c) => (
            <div
              key={c.id}
              className={`sidebar-chat-item${c.id === conversationId ? " active" : ""}`}
            >
              <NavLink
                to={`/chat/${c.id}`}
                className="sidebar-chat-link"
                title={c.title || "New chat"}
              >
                {c.title || "New chat"}
              </NavLink>
              <button
                className="conversation-delete"
                type="button"
                title="Delete chat"
                onClick={() => handleDeleteChat(c.id)}
              >
                ×
              </button>
            </div>
          ))}
        </div>

        <Pagination
          page={page}
          pageCount={pageCount}
          rangeStart={rangeStart}
          rangeEnd={rangeEnd}
          total={total}
          onPrev={prev}
          onNext={next}
          compact
        />
      </div>

      <div className="sidebar-footer">
        <div>
          <div className="small muted">{email ?? "Signed in"}</div>
          <button
            className="btn btn-secondary small"
            style={{ marginTop: 10 }}
            type="button"
            onClick={handleLogout}
          >
            Sign out
          </button>
        </div>
      </div>
    </aside>
  );
}
