export type ListRow<T> =
  | { kind: 'header'; key: string; label: string; count: number }
  | { kind: 'item'; key: string; item: T };

export interface ListSection<T> {
  key: string;
  label: string;
  items: readonly T[];
}

/** Sections flatten to one row sequence so headers virtualize with their items; empty sections contribute nothing. */
export const listRowsFromSections = <T>(
  sections: readonly ListSection<T>[],
  getItemKey: (item: T) => string
): ListRow<T>[] => {
  const rows: ListRow<T>[] = [];

  for (const section of sections) {
    if (section.items.length === 0) {
      continue;
    }

    rows.push({ count: section.items.length, key: `header:${section.key}`, kind: 'header', label: section.label });

    for (const item of section.items) {
      rows.push({ item, key: getItemKey(item), kind: 'item' });
    }
  }

  return rows;
};

export const listRowsFromItems = <T>(items: readonly T[], getItemKey: (item: T) => string): ListRow<T>[] =>
  items.map((item) => ({ item, key: getItemKey(item), kind: 'item' }));
