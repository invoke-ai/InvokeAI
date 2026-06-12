// Matches a "[category]rest" prefix at the start of a model name. Used by the model picker to extract a
// user-defined category for grouping/sorting. The underlying model name (e.g. in the model manager) is unchanged.
const CATEGORY_PATTERN = /^\s*\[([^\]]+)\]\s*(.*)$/;

type ParsedModelCategory = {
  category: string | null;
  displayName: string;
};

export const parseCategoryFromName = (name: string): ParsedModelCategory => {
  const match = name.match(CATEGORY_PATTERN);
  if (match && match[1] !== undefined) {
    const category = match[1].trim();
    const rest = (match[2] ?? '').trim();
    if (category) {
      return { category, displayName: rest || name };
    }
  }
  return { category: null, displayName: name };
};
