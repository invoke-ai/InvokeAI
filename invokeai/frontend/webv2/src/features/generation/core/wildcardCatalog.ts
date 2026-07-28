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

import { searchCatalog } from './catalogSearch';

export interface WildcardListEntry {
  name: string;
  values: readonly string[];
}

export interface WildcardGroup<T> {
  /** `null` for names with no `/` — those render without a header. */
  label: string | null;
  wildcards: T[];
}

/**
 * Name matches first, ranked fuzzily, then wildcards whose *values* contain the
 * query — `searchCatalog`'s rule, which the template and history lists follow
 * too. Values are this record's prose: you remember that "cyberpunk" is in one
 * of these lists long before you remember which.
 */
const getWildcardProse = (wildcard: WildcardListEntry): readonly string[] => wildcard.values;

export const filterWildcards = <T extends WildcardListEntry>(wildcards: readonly T[], query: string): T[] =>
  searchCatalog(wildcards, query, getWildcardProse);

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
