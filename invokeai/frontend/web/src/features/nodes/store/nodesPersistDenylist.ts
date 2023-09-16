import { NodesState } from './types';

/**
 * Nodes slice persist denylist
 */
export const nodesPersistDenylist: (keyof NodesState)[] = [
  'nodeTemplates',
  'connectionStartParams',
  'currentConnectionFieldType',
  'selectedNodes',
  'selectedEdges',
  'isReady',
  'nodesToCopy',
  'edgesToCopy',
];
