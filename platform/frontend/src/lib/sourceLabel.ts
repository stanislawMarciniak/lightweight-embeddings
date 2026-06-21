/** Human-readable provenance label shown under assistant replies. */
export function responseSourceLabel(
  source?: string | null,
  documentName?: string | null
): string | undefined {
  switch (source) {
    case "faq":
      return "answer from FAQ";
    case "multiple_faq":
      return "answer from multiple FAQ";
    case "document":
      return documentName
        ? `answer from document ${documentName}`
        : "answer from document";
    default:
      return undefined;
  }
}
