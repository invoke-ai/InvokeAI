/**
 * Finding one wildcard among many, and giving `a/b` names the shape their author
 * meant by them.
 *
 * Nested names have always been legal — `WILDCARD_NAME_SOURCE` in
 * `./dynamicPrompts` allows `animals/dogs` — but nothing has ever surfaced the
 * nesting, so a namespaced catalog reads as a flat list of unrelated rows. Both
 * helpers here are pure and ordering-preserving: the caller decides what order
 * the wildcards arrive in (catalog order normally, match order while searching)
 * and grouping never reshuffles it.
 */

import fuzzysort from 'fuzzysort';

export interface WildcardListEntry {
  name: string;
  values: readonly string[];
}

/** Matches below this fuzzysort score (0–1) are noise, not results. */
const NAME_MATCH_THRESHOLD = 0.2;

export interface WildcardGroup<T> {
  /** `null` for names with no `/` — those render without a header. */
  label: string | null;
  wildcards: T[];
}

/**
 * Name matches first, ranked fuzzily, then wildcards whose *values* contain the
 * query.
 *
 * The split is deliberate. Names are short slugs, where fuzzy matching earns its
 * keep (`adg` finds `animals/dogs`); values are free text, where a fuzzy match
 * would fire on almost anything, so they take a plain substring test. Searching
 * values at all matters because a wildcard is mostly its values — you remember
 * that "cyberpunk" is in one of these lists long before you remember which.
 *
 * Values are scanned in full rather than sampled: stopping early would silently
 * hide a match in a long list, which is worse than the scan being slow. `.some`
 * exits on the first hit.
 */
export const filterWildcards = <T extends WildcardListEntry>(wildcards: readonly T[], query: string): T[] => {
  const trimmed = query.trim();

  if (trimmed.length === 0) {
    return [...wildcards];
  }

  const nameMatches = fuzzysort.go(trimmed, wildcards, {
    key: (wildcard: T) => wildcard.name,
    threshold: NAME_MATCH_THRESHOLD,
  });
  const matched = new Set<T>(nameMatches.map((result) => result.obj));
  const results = nameMatches.map((result) => result.obj);
  const needle = trimmed.toLowerCase();

  for (const wildcard of wildcards) {
    if (!matched.has(wildcard) && wildcard.values.some((value) => value.toLowerCase().includes(needle))) {
      results.push(wildcard);
    }
  }

  return results;
};

/**
 * Splits on the *first* `/` only. `characters/fantasy/elves` groups under
 * `characters`, not `characters/fantasy` — one level of headers keeps the panel
 * scannable, and a deeper tree would need collapsing to stay usable in a 14rem
 * box.
 *
 * Top-level names come first and unlabelled, so a catalog that uses no `/` at
 * all renders exactly as it did before grouping existed. Labelled groups follow
 * in first-appearance order, which is match order during a search — the best hit
 * stays at the top rather than being sorted away.
 */
export const groupWildcardsByPrefix = <T extends { name: string }>(wildcards: readonly T[]): WildcardGroup<T>[] => {
  const ungrouped: T[] = [];
  const groups = new Map<string, T[]>();

  for (const wildcard of wildcards) {
    const separatorIndex = wildcard.name.indexOf('/');

    if (separatorIndex <= 0) {
      ungrouped.push(wildcard);
      continue;
    }

    const label = wildcard.name.slice(0, separatorIndex);
    const existing = groups.get(label);

    if (existing) {
      existing.push(wildcard);
    } else {
      groups.set(label, [wildcard]);
    }
  }

  return [
    ...(ungrouped.length > 0 ? [{ label: null, wildcards: ungrouped }] : []),
    ...[...groups].map(([label, grouped]) => ({ label, wildcards: grouped })),
  ];
};
