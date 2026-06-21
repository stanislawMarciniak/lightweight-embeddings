export type MessageRole = "user" | "assistant";

interface MessageProps {
  role: MessageRole;
  content: string;
  sourceLabel?: string;
}

export function Message({ role, content, sourceLabel }: MessageProps) {
  const isUser = role === "user";
  return (
    <div className={`message-row ${isUser ? "message-user" : "message-assistant"}`}>
      <div className="message-content">
        <div
          className={`message-bubble ${
            isUser ? "message-bubble-user" : "message-bubble-assistant"
          }`}
        >
          {content}
        </div>
        {!isUser && sourceLabel ? (
          <div className="message-source">{sourceLabel}</div>
        ) : null}
      </div>
    </div>
  );
}
