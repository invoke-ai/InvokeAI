/**
 * One search rule for every saved-prompt catalog: fuzzy on the short name,
 * substring on the long prose, name matches first.
 *
 * The three lists that hold saved things — wildcards, prompt templates, prompt
 * history — each grew their own filter, and the two that sit side by side in
 * the same popover disagreed about what typing means: a wildcard was found by
 * name fuzzily (`adg` finds `animals/dogs`), a template only where its name
 * contained the query letter for letter. Same box, same gesture, two answers.
 *
 * The split between the two kinds of field is the part worth keeping, and it is
 * why one rule fits all three. Names are short slugs, where fuzzy matching earns
 * its keep; prose is free text, where a fuzzy match fires on almost anything, so
 * it takes a plain substring test. Searching the prose at all matters because a
 * record is mostly its prose — you remember that "cyberpunk" is in one of these
 * lists long before you remember which.
 *
 * Prose is scanned in full rather than sampled: stopping early would silently
 * hide a match in a long list, which is worse than the scan being slow.
 *
 * The inline `__` autocomplete deliberately does *not* come through here. It
 * completes a word while it is being typed, where anything looser than a
 * substring on the one visible field inserts text nobody asked for.
 */

import fuzzysort from 'fuzzysort';

/** Matches below this fuzzysort score (0–1) are noise, not results. */
const NAME_MATCH_THRESHOLD = 0.2;

type ProseOf<T> = (item: T) => readonly string[];

/**
 * Lowercased prose, derived once per record rather than once per keystroke.
 *
 * Lowercasing on the fly was the whole cost of searching a large catalog: 200
 * wildcards of 200 values allocated 40 000 strings per character typed, and the
 * search box is not debounced. Keyed by record identity, so an edited record —
 * a new object every time, as the query cache hands them back — derives afresh.
 *
 * Keyed by the accessor too, so two callers reading different fields off the
 * same record can never read each other's lowercased copy.
 *
 * Fields stay separate rather than joining into one string: a query is free to
 * contain whatever the user typed, and no separator both survives that and
 * stops a match from spanning the end of one value and the start of the next.
 */
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

/**
 * Records matching on their prose alone, in the order given. For lists whose
 * entries have no name of their own — prompt history is the prompts.
 */
export const searchProse = <T extends object>(items: readonly T[], query: string, getProse: ProseOf<T>): T[] => {
  const needle = query.trim().toLowerCase();

  if (needle.length === 0) {
    return [...items];
  }

  return items.filter((item) => matchesProse(item, getProse, needle));
};

/**
 * Name matches first, ranked fuzzily, then records whose prose contains the
 * query, in the order given.
 *
 * Ordering is otherwise preserved, so a caller that groups or sections the
 * result keeps the ranking: the best hit stays at the top rather than being
 * sorted away.
 */
export const searchCatalog = <T extends { name: string }>(
  items: readonly T[],
  query: string,
  getProse: ProseOf<T>
): T[] => {
  const trimmed = query.trim();

  if (trimmed.length === 0) {
    return [...items];
  }

  // `key: 'name'` rather than an accessor: fuzzysort reads `obj[key]` before it
  // considers calling it, so a function key is first used as a property key —
  // stringifying the function's source once per record per keystroke. Plain
  // names also hit fuzzysort's own prepared-target cache.
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
