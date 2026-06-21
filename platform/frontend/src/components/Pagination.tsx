interface PaginationProps {
  page: number;
  pageCount: number;
  rangeStart: number;
  rangeEnd: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
  /** Compact variant for tight spaces (e.g. the sidebar). */
  compact?: boolean;
  /** Noun for the range label, e.g. "FAQs" or "documents". */
  label?: string;
}

export function Pagination({
  page,
  pageCount,
  rangeStart,
  rangeEnd,
  total,
  onPrev,
  onNext,
  compact = false,
  label
}: PaginationProps) {
  if (pageCount <= 1) return null;

  return (
    <div className={`pagination${compact ? " pagination-compact" : ""}`}>
      <button
        className="btn btn-secondary small"
        type="button"
        onClick={onPrev}
        disabled={page <= 1}
      >
        Prev
      </button>
      <span className="pagination-info muted small">
        {compact
          ? `${page} / ${pageCount}`
          : `${rangeStart}\u2013${rangeEnd} of ${total}${label ? ` ${label}` : ""}`}
      </span>
      <button
        className="btn btn-secondary small"
        type="button"
        onClick={onNext}
        disabled={page >= pageCount}
      >
        Next
      </button>
    </div>
  );
}
