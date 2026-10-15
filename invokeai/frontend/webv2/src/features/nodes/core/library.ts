import type { NodePackInfo } from './catalog';

export type NodePackSortField = 'name' | 'nodeCount' | 'path';

export interface NodePackFilters {
  searchTerm: string;
  /** Only packs that registered no nodes — import failed or a restart is pending. */
  problemsOnly: boolean;
  sortField: NodePackSortField;
  sortDirection: 'asc' | 'desc';
}

export const DEFAULT_NODE_PACK_FILTERS: NodePackFilters = {
  problemsOnly: false,
  searchTerm: '',
  sortDirection: 'asc',
  sortField: 'name',
};

/**
 * Zero nodes indicates failed import or pending reload/restart because counts come from the live invocation
 * registry.
 */
export const isProblemPack = (pack: Pick<NodePackInfo, 'nodeCount'>): boolean => pack.nodeCount === 0;

const compareBySortField = (a: NodePackInfo, b: NodePackInfo, field: NodePackSortField): number => {
  switch (field) {
    case 'name':
      return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
    case 'nodeCount':
      return a.nodeCount - b.nodeCount;
    case 'path':
      return a.path.localeCompare(b.path);
  }
};

export const filterNodePacks = (packs: NodePackInfo[], filters: NodePackFilters): NodePackInfo[] => {
  const query = filters.searchTerm.trim().toLowerCase();
  const directionFactor = filters.sortDirection === 'desc' ? -1 : 1;

  return packs
    .filter(
      (pack) =>
        (query === '' || pack.name.toLowerCase().includes(query) || pack.path.toLowerCase().includes(query)) &&
        (!filters.problemsOnly || isProblemPack(pack))
    )
    .sort((a, b) => compareBySortField(a, b, filters.sortField) * directionFactor);
};
