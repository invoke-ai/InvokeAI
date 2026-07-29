import fuzzysort from 'fuzzysort';

const NAME_MATCH_THRESHOLD = 0.2;

type ProseOf<T> = (item: T) => readonly string[];

// Both record and accessor identity are cache keys; callers may read different fields from one record.
const proseCaches = new WeakMap<object, WeakMap<object, readonly string[]>>();

const matchesProse = <T extends object>(item: T, getProse: ProseOf<T>, needle: string): boolean => {
  let cache = proseCaches.get(getProse);

  if (!cache) {
    cache = new WeakMap<object, readonly string[]>();
    proseCaches.set(getProse, cache);
  }

  let lowercased = cache.get(item);

  if (lowercased === undefined) {
    lowercased = getProse(item).map((field) => field.toLowerCase());
    cache.set(item, lowercased);
  }

  return lowercased.some((field) => field.includes(needle));
};

export const searchProse = <T extends object>(items: readonly T[], query: string, getProse: ProseOf<T>): T[] => {
  const needle = query.trim().toLowerCase();

  if (needle.length === 0) {
    return [...items];
  }

  return items.filter((item) => matchesProse(item, getProse, needle));
};

export const searchCatalog = <T extends { name: string }>(
  items: readonly T[],
  query: string,
  getProse: ProseOf<T>
): T[] => {
  const trimmed = query.trim();

  if (trimmed.length === 0) {
    return [...items];
  }

  const nameMatches = fuzzysort.go(trimmed, items, { key: 'name', threshold: NAME_MATCH_THRESHOLD });
  const named = new Set<T>();
  const results: T[] = [];

  for (const match of nameMatches) {
    named.add(match.obj);
    results.push(match.obj);
  }

  const needle = trimmed.toLowerCase();

  for (const item of items) {
    if (!named.has(item) && matchesProse(item, getProse, needle)) {
      results.push(item);
    }
  }

  return results;
};
