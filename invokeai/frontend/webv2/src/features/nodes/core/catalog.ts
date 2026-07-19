export interface NodePackInfo {
  name: string;
  path: string;
  nodeCount: number;
  nodeTypes: string[];
}

export interface NodePackCatalog {
  customNodesPath: string;
  nodePacks: NodePackInfo[];
}

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

export const normalizeNodePackInfo = (value: unknown): NodePackInfo | null => {
  const record = asRecord(value);

  if (!record || typeof record.name !== 'string' || typeof record.path !== 'string') {
    return null;
  }

  return {
    name: record.name,
    nodeCount: typeof record.node_count === 'number' && Number.isFinite(record.node_count) ? record.node_count : 0,
    nodeTypes: Array.isArray(record.node_types)
      ? record.node_types.filter((nodeType): nodeType is string => typeof nodeType === 'string')
      : [],
    path: record.path,
  };
};

export const normalizeNodePackCatalog = (value: unknown): NodePackCatalog => {
  const record = asRecord(value);
  const nodePacks = Array.isArray(record?.node_packs)
    ? record.node_packs.map(normalizeNodePackInfo).filter((pack): pack is NodePackInfo => pack !== null)
    : [];

  return {
    customNodesPath: typeof record?.custom_nodes_path === 'string' ? record.custom_nodes_path : '',
    nodePacks,
  };
};
