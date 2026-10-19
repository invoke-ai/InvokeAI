import { useDeferredValue, useMemo, useState } from 'react';

/** Last path/URL segment: install sources display and filter by file name. */
export const sourceFileName = (source: string): string => source.split(/[\\/]/).at(-1) ?? source;

const HF_RESOLVE_PATH = /\/resolve\/[^/]+\/(.+)$/;

/**
 * Where a source sits under its repo or scan root: the part that tells same-named files apart. Null when that is
 * just the file name, which the row already shows.
 */
export const sourceLocation = (source: string, root?: string): string | null => {
  const inRepo = HF_RESOLVE_PATH.exec(source)?.[1];
  const trimmedRoot = root?.replace(/[\\/]+$/, '');
  const relative =
    inRepo ??
    (trimmedRoot && source.startsWith(trimmedRoot) ? source.slice(trimmedRoot.length).replace(/^[\\/]+/, '') : source);

  return relative === sourceFileName(source) ? null : relative;
};

/** Pass a module-stable sourceOf so deferred filename filtering stays keyed on items. */
export const useSourceNameFilter = <Item>(items: readonly Item[], sourceOf: (item: Item) => string) => {
  const [filter, setFilter] = useState('');
  const deferredFilter = useDeferredValue(filter);

  const filteredItems = useMemo(() => {
    const term = deferredFilter.trim().toLowerCase();

    if (!term) {
      return items;
    }

    return items.filter((item) => sourceFileName(sourceOf(item)).toLowerCase().includes(term));
  }, [deferredFilter, items, sourceOf]);

  return { filter, filteredItems, setFilter };
};
