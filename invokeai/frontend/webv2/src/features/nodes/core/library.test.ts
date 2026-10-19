import { describe, expect, it } from 'vitest';

import type { NodePackInfo } from './catalog';

import { DEFAULT_NODE_PACK_FILTERS, filterNodePacks, isProblemPack } from './library';

const pack = (name: string, nodeCount: number, path = `/custom_nodes/${name}`): NodePackInfo => ({
  name,
  nodeCount,
  nodeTypes: [],
  path,
});

const library: NodePackInfo[] = [pack('zeta-pack', 5), pack('alpha-pack', 0), pack('midway', 12, '/elsewhere/midway')];

describe('filterNodePacks', () => {
  it('searches name and path case-insensitively', () => {
    expect(
      filterNodePacks(library, { ...DEFAULT_NODE_PACK_FILTERS, searchTerm: 'ZETA' }).map((entry) => entry.name)
    ).toEqual(['zeta-pack']);
    expect(
      filterNodePacks(library, { ...DEFAULT_NODE_PACK_FILTERS, searchTerm: 'elsewhere' }).map((entry) => entry.name)
    ).toEqual(['midway']);
  });

  it('isolates problem packs', () => {
    expect(
      filterNodePacks(library, { ...DEFAULT_NODE_PACK_FILTERS, problemsOnly: true }).map((entry) => entry.name)
    ).toEqual(['alpha-pack']);
  });

  it('sorts by every field in both directions', () => {
    const names = (filters: Partial<typeof DEFAULT_NODE_PACK_FILTERS>) =>
      filterNodePacks(library, { ...DEFAULT_NODE_PACK_FILTERS, ...filters }).map((entry) => entry.name);

    expect(names({ sortField: 'name' })).toEqual(['alpha-pack', 'midway', 'zeta-pack']);
    expect(names({ sortDirection: 'desc', sortField: 'name' })).toEqual(['zeta-pack', 'midway', 'alpha-pack']);
    expect(names({ sortField: 'nodeCount' })).toEqual(['alpha-pack', 'zeta-pack', 'midway']);
    expect(names({ sortDirection: 'desc', sortField: 'nodeCount' })).toEqual(['midway', 'zeta-pack', 'alpha-pack']);
    expect(names({ sortField: 'path' })).toEqual(['alpha-pack', 'zeta-pack', 'midway']);
  });

  it('handles an empty library', () => {
    expect(filterNodePacks([], DEFAULT_NODE_PACK_FILTERS)).toEqual([]);
  });
});

describe('isProblemPack', () => {
  it('flags only zero-node packs', () => {
    expect(isProblemPack({ nodeCount: 0 })).toBe(true);
    expect(isProblemPack({ nodeCount: 3 })).toBe(false);
  });
});
