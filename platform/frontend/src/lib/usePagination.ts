import { useEffect, useMemo, useState } from "react";

export interface PaginationState<T> {
  page: number;
  pageCount: number;
  total: number;
  pageSize: number;
  pageItems: T[];
  rangeStart: number;
  rangeEnd: number;
  setPage: (p: number) => void;
  next: () => void;
  prev: () => void;
}

/**
 * Client-side pagination: slices `items` into pages of `pageSize` (default 10).
 * Keeps the current page valid when the underlying list grows or shrinks.
 */
export function usePagination<T>(items: T[], pageSize = 10): PaginationState<T> {
  const [page, setPage] = useState(1);
  const total = items.length;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));

  useEffect(() => {
    if (page > pageCount) setPage(pageCount);
  }, [page, pageCount]);

  const safePage = Math.min(page, pageCount);
  const start = (safePage - 1) * pageSize;

  const pageItems = useMemo(
    () => items.slice(start, start + pageSize),
    [items, start, pageSize]
  );

  return {
    page: safePage,
    pageCount,
    total,
    pageSize,
    pageItems,
    rangeStart: total === 0 ? 0 : start + 1,
    rangeEnd: Math.min(start + pageSize, total),
    setPage,
    next: () => setPage((p) => Math.min(pageCount, p + 1)),
    prev: () => setPage((p) => Math.max(1, p - 1))
  };
}
