import { ChangeEvent, useEffect, useState } from "react";
import { Sidebar } from "../components/Sidebar";
import { Pagination } from "../components/Pagination";
import { usePagination } from "../lib/usePagination";
import { DocumentItem, apiDeleteDocument, apiGetDocuments, apiUploadDocument } from "../lib/api";

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { pageItems, page, pageCount, rangeStart, rangeEnd, total, next, prev } = usePagination(
    documents,
    10
  );

  async function loadDocuments() {
    try {
      const data = await apiGetDocuments();
      setDocuments(data);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Failed to load documents", err);
    }
  }

  useEffect(() => {
    loadDocuments();
  }, []);

  async function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);

    if (!["text/plain", "application/pdf"].includes(file.type)) {
      setError("Only .txt and .pdf files are supported");
      return;
    }

    setUploading(true);
    try {
      const res = await apiUploadDocument(file);
      setDocuments((prev) => [...prev, res.document]);
    } catch (err: any) {
      setError(err?.message ?? "Failed to upload document");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function handleDelete(id: number) {
    try {
      await apiDeleteDocument(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Failed to delete document", err);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-content">
        <div className="page-header">
          <div className="page-title">Documents</div>
          <div className="muted small">Upload .txt and .pdf files for semantic search.</div>
        </div>

        <div className="card">
          <div className="field-group">
            <label className="field-label" htmlFor="doc-upload">
              Upload document
            </label>
            <input
              id="doc-upload"
              type="file"
              accept=".txt,application/pdf"
              onChange={handleFileChange}
            />
            <div className="muted small">Each sentence is embedded and indexed per user.</div>
          </div>

          {error && <div className="error-text">{error}</div>}

          <div className="documents-list">
            {documents.length === 0 && (
              <div className="muted small">No documents uploaded yet.</div>
            )}
            {pageItems.map((doc) => (
              <div key={doc.id} className="document-item">
                <div>
                  <div>{doc.filename}</div>
                  <div className="muted small">
                    Stored at: <code>{doc.storage_path}</code>
                  </div>
                </div>
                <button
                  className="btn btn-danger small"
                  type="button"
                  disabled={uploading}
                  onClick={() => handleDelete(doc.id)}
                >
                  Delete
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
            label="documents"
          />
        </div>
      </main>
    </div>
  );
}

