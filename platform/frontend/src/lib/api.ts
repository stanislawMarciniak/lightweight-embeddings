import { supabase } from "./supabaseClient";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

const apiBaseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? DEFAULT_API_BASE_URL;

async function getAccessToken(): Promise<string | null> {
  const {
    data: { session }
  } = await supabase.auth.getSession();
  return session?.access_token ?? null;
}

async function request<T>(
  path: string,
  options: RequestInit & { auth?: boolean } = {}
): Promise<T> {
  const url = `${apiBaseUrl}${path}`;
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(options.headers ?? {})
  };

  if (options.auth) {
    const token = await getAccessToken();
    if (!token) {
      throw new Error("Not authenticated");
    }
    (headers as Record<string, string>)["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(url, {
    ...options,
    headers
  });

  if (!res.ok) {
    let errorText = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        errorText = body.detail;
      }
    } catch {
      // ignore parse error
    }
    throw new Error(errorText);
  }

  if (res.status === 204) {
    return {} as T;
  }

  return (await res.json()) as T;
}

export interface ChatItem {
  id: string;
  user_id: string;
  conversation_id: string | null;
  message: string;
  response: string;
  response_source?: string | null;
  response_document_name?: string | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string | null;
}

export interface FAQItem {
  id: string;
  user_id: string;
  question: string;
  answer: string;
  embedding: number[];
  created_at: string;
}

export interface DocumentItem {
  id: number;
  user_id: string;
  filename: string;
  storage_path: string;
  created_at: string;
}

export async function apiSendChat(message: string, conversationId?: string | null) {
  return request<{
    message: string;
    response: string;
    conversation_id: string;
    response_source?: string | null;
    response_document_name?: string | null;
  }>("/chat", {
    method: "POST",
    auth: true,
    body: JSON.stringify({ message, conversation_id: conversationId ?? null })
  });
}

export async function apiGetChats() {
  return request<ChatItem[]>("/chats", {
    method: "GET",
    auth: true
  });
}

export async function apiGetConversations() {
  return request<Conversation[]>("/conversations", { method: "GET", auth: true });
}

export async function apiCreateConversation(title?: string) {
  return request<Conversation>("/conversations", {
    method: "POST",
    auth: true,
    body: JSON.stringify({ title: title ?? null })
  });
}

export async function apiRenameConversation(id: string, title: string) {
  return request<Conversation>(`/conversations/${id}`, {
    method: "PATCH",
    auth: true,
    body: JSON.stringify({ title })
  });
}

export async function apiDeleteConversation(id: string) {
  return request<{ detail: string }>(`/conversations/${id}`, {
    method: "DELETE",
    auth: true
  });
}

export async function apiGetConversationMessages(id: string) {
  return request<ChatItem[]>(`/conversations/${id}/messages`, { method: "GET", auth: true });
}

export async function apiGetFaq() {
  return request<FAQItem[]>("/faq", {
    method: "GET",
    auth: true
  });
}

export async function apiCreateFaq(question: string, answer: string) {
  return request<FAQItem>("/faq", {
    method: "POST",
    auth: true,
    body: JSON.stringify({ question, answer })
  });
}

export async function apiDeleteFaq(id: string) {
  return request<{ detail: string }>(`/faq/${id}`, {
    method: "DELETE",
    auth: true
  });
}

export interface FAQImportResult {
  created: number;
  skipped: number;
  items: FAQItem[];
}

export async function apiImportFaq(file: File) {
  const token = await getAccessToken();
  if (!token) {
    throw new Error("Not authenticated");
  }
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${apiBaseUrl}/faq/import`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: formData
  });

  if (!res.ok) {
    let errorText = `Import failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) errorText = body.detail;
    } catch {
      // ignore parse error
    }
    throw new Error(errorText);
  }
  return (await res.json()) as FAQImportResult;
}

export interface AutocompleteSuggestion {
  id: string;
  question: string;
  score: number;
}

export async function apiAutocomplete(q: string, limit = 2, signal?: AbortSignal) {
  return request<{ q: string; suggestions: AutocompleteSuggestion[] }>(
    `/faq/autocomplete?q=${encodeURIComponent(q)}&limit=${limit}`,
    { method: "GET", auth: true, signal }
  );
}

export async function apiFaqSuggestions(pageContext: string, limit = 3) {
  return request<{ page_context: string; suggestions: AutocompleteSuggestion[] }>(
    `/faq/suggestions?page_context=${encodeURIComponent(pageContext)}&limit=${limit}`,
    { method: "GET", auth: true }
  );
}

export async function apiGetDocuments() {
  return request<DocumentItem[]>("/documents", {
    method: "GET",
    auth: true
  });
}

export async function apiDeleteDocument(id: number) {
  return request<{ detail: string }>(`/documents/${id}`, {
    method: "DELETE",
    auth: true
  });
}

export async function apiUploadDocument(file: File) {
  const token = await getAccessToken();
  if (!token) {
    throw new Error("Not authenticated");
  }

  const url = `${apiBaseUrl}/documents/upload`;
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`
    },
    body: formData
  });

  if (!res.ok) {
    let errorText = `Upload failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        errorText = body.detail;
      }
    } catch {
      // ignore parse error
    }
    throw new Error(errorText);
  }

  return (await res.json()) as {
    document: DocumentItem;
    chunks_created: number;
  };
}

