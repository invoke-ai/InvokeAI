import type { NodesState } from './types';

/**
 * Nodes slice persist denylist
 */
export const nodesPersistDenylist: (keyof NodesState)[] = [
  'connectionStartParams',
  'connectionStartFieldType',
  'selectedNodes',
  'selectedEdges',
  'isReady',
  'nodesToCopy',
  'edgesToCopy',
  'connectionMade',
  'modifyingEdge',
  'addNewNodePosition',
];
