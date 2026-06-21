import { useEffect, useRef, useState } from "react";
import { Message, MessageRole } from "./Message";
import { apiAutocomplete, AutocompleteSuggestion } from "../lib/api";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  sourceLabel?: string;
}

interface ChatProps {
  messages: ChatMessage[];
  input: string;
  setInput: (value: string) => void;
  onSend: () => void;
  sending: boolean;
}

const DEBOUNCE_MS = 120;
const MIN_CHARS = 2;

export function Chat({ messages, input, setInput, onSend, sending }: ChatProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // --- semantic autocomplete state ---
  const [suggestions, setSuggestions] = useState<AutocompleteSuggestion[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const debounceRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // when the user accepts a suggestion we suppress the next fetch
  const suppressRef = useRef(false);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Debounced autocomplete on every keystroke.
  useEffect(() => {
    if (suppressRef.current) {
      suppressRef.current = false;
      return;
    }
    if (debounceRef.current) window.clearTimeout(debounceRef.current);

    const q = input.trim();
    if (q.length < MIN_CHARS || sending) {
      setSuggestions([]);
      setOpen(false);
      return;
    }

    debounceRef.current = window.setTimeout(async () => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const res = await apiAutocomplete(q, 2, controller.signal);
        setSuggestions(res.suggestions);
        setOpen(res.suggestions.length > 0);
        setActive(-1);
      } catch {
        // aborted or failed -> keep silent (autocomplete is best-effort)
      }
    }, DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current);
    };
  }, [input, sending]);

  function acceptSuggestion(s: AutocompleteSuggestion) {
    suppressRef.current = true;
    setInput(s.question);
    setSuggestions([]);
    setOpen(false);
    setActive(-1);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (open && suggestions.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive((a) => (a + 1) % suggestions.length);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive((a) => (a <= 0 ? suggestions.length - 1 : a - 1));
        return;
      }
      if (e.key === "Escape") {
        setOpen(false);
        return;
      }
      if (e.key === "Enter" && !e.shiftKey && active >= 0) {
        e.preventDefault();
        acceptSuggestion(suggestions[active]);
        return;
      }
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!sending && input.trim()) {
        setOpen(false);
        onSend();
      }
    }
  }

  return (
    <div className="chat-container card">
      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="muted small">
            Ask a question and I will search your FAQs and documents, then fall back to OpenAI
            using document context.
          </div>
        )}
        {messages.map((m) => (
          <Message key={m.id} role={m.role} content={m.content} sourceLabel={m.sourceLabel} />
        ))}
      </div>
      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          if (!sending && input.trim()) {
            setOpen(false);
            onSend();
          }
        }}
      >
        <div className="autocomplete-wrap">
          {open && suggestions.length > 0 && (
            <ul className="autocomplete-dropdown" role="listbox">
              {suggestions.map((s, i) => (
                <li
                  key={s.id}
                  role="option"
                  aria-selected={i === active}
                  className={`autocomplete-item${i === active ? " active" : ""}`}
                  // onMouseDown (not onClick) so it fires before textarea blur
                  onMouseDown={(e) => {
                    e.preventDefault();
                    acceptSuggestion(s);
                  }}
                  onMouseEnter={() => setActive(i)}
                >
                  <span className="autocomplete-icon">⌕</span>
                  <span className="autocomplete-text">{s.question}</span>
                </li>
              ))}
            </ul>
          )}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={() => window.setTimeout(() => setOpen(false), 120)}
            onFocus={() => suggestions.length > 0 && setOpen(true)}
            placeholder="Ask a question..."
          />
        </div>
        <button className="btn btn-primary" type="submit" disabled={sending || !input.trim()}>
          {sending ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
}
