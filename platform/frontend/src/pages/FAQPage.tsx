import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import { Sidebar } from "../components/Sidebar";
import { Pagination } from "../components/Pagination";
import { usePagination } from "../lib/usePagination";
import { FAQItem, apiCreateFaq, apiDeleteFaq, apiGetFaq, apiImportFaq } from "../lib/api";

export function FAQPage() {
  const [faqs, setFaqs] = useState<FAQItem[]>([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [importMsg, setImportMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { pageItems, page, pageCount, rangeStart, rangeEnd, total, next, prev } = usePagination(
    faqs,
    10
  );

  async function loadFaqs() {
    try {
      const data = await apiGetFaq();
      setFaqs(data);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Failed to load FAQ", err);
    }
  }

  useEffect(() => {
    loadFaqs();
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const created = await apiCreateFaq(question, answer);
      setFaqs((prev) => [...prev, created]);
      setQuestion("");
      setAnswer("");
    } catch (err: any) {
      setError(err?.message ?? "Failed to create FAQ");
    } finally {
      setLoading(false);
    }
  }

  async function handleImport(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setImportMsg(null);
    setImporting(true);
    try {
      const result = await apiImportFaq(file);
      setImportMsg(
        `Imported ${result.created} FAQ${result.created === 1 ? "" : "s"}` +
          (result.skipped > 0 ? ` (${result.skipped} skipped)` : "")
      );
      await loadFaqs();
    } catch (err: any) {
      setError(err?.message ?? "Failed to import FAQ");
    } finally {
      setImporting(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleDelete(id: string) {
    try {
      await apiDeleteFaq(id);
      setFaqs((prev) => prev.filter((f) => f.id !== id));
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Failed to delete FAQ", err);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-content">
        <div className="page-header">
          <div className="page-title">FAQ</div>
          <div className="muted small">Maintain per-user FAQ entries with semantic search.</div>
        </div>

        <div className="card">
          <div className="faq-import">
            <div>
              <div className="field-label">Bulk import</div>
              <div className="muted small">
                Upload a .json file: a list of {`{ "question": "...", "answer": "..." }`} (or{" "}
                {`{ "faqs": [...] }`}).
              </div>
            </div>
            <div className="faq-import-actions">
              <input
                ref={fileInputRef}
                type="file"
                accept="application/json,.json"
                style={{ display: "none" }}
                onChange={handleImport}
              />
              <button
                className="btn btn-secondary"
                type="button"
                disabled={importing}
                onClick={() => fileInputRef.current?.click()}
              >
                {importing ? "Importing..." : "Import JSON"}
              </button>
            </div>
          </div>
          {importMsg && <div className="small" style={{ color: "#34d399" }}>{importMsg}</div>}
        </div>

        <div className="card">
          <form onSubmit={handleSubmit}>
            <div className="field-group">
              <label className="field-label" htmlFor="faq-question">
                Question
              </label>
              <input
                id="faq-question"
                className="field-input"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                required
              />
            </div>

            <div className="field-group">
              <label className="field-label" htmlFor="faq-answer">
                Answer
              </label>
              <textarea
                id="faq-answer"
                className="field-textarea"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                required
              />
            </div>

            {error && <div className="error-text">{error}</div>}

            <button className="btn btn-primary" type="submit" disabled={loading}>
              {loading ? "Saving..." : "Add FAQ"}
            </button>
          </form>

          <div className="faq-list">
            {faqs.length === 0 && <div className="muted small">No FAQ entries yet.</div>}
            {pageItems.map((faq) => (
              <div key={faq.id} className="faq-item">
                <div className="faq-item-header">
                  <div className="small muted">
                    <span className="tag">Q</span> {faq.question}
                  </div>
                  <button
                    className="btn btn-danger small"
                    type="button"
                    onClick={() => handleDelete(faq.id)}
                  >
                    Delete
                  </button>
                </div>
                <div className="small" style={{ marginTop: "0.35rem" }}>
                  <span className="tag">A</span> {faq.answer}
                </div>
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
            label="FAQs"
          />
        </div>
      </main>
    </div>
  );
}

