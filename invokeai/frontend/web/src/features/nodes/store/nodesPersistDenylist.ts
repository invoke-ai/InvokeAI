import { NodesState } from './types';

/**
 * Nodes slice persist denylist
 */
export const nodesPersistDenylist: (keyof NodesState)[] = [
  'schema',
  'nodeTemplates',
  'connectionStartParams',
  'currentConnectionFieldType',
  'selectedNodes',
  'selectedEdges',
];
