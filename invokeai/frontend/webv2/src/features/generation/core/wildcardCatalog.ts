import { searchCatalog } from './catalogSearch';

export interface WildcardListEntry {
  name: string;
  values: readonly string[];
}

export interface WildcardGroup<T> {
  label: string | null;
  wildcards: T[];
}

// Keep this accessor stable: catalogSearch keys its prose cache by accessor identity.
const getWildcardProse = (wildcard: WildcardListEntry): readonly string[] => wildcard.values;

export const filterWildcards = <T extends WildcardListEntry>(wildcards: readonly T[], query: string): T[] =>
  searchCatalog(wildcards, query, getWildcardProse);

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
