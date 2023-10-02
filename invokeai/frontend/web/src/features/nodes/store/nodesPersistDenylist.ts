import { NodesState } from './types';

/**
 * Nodes slice persist denylist
 */
export const nodesPersistDenylist: (keyof NodesState['present'])[] = [
  'nodeTemplates',
  'connectionStartParams',
  'currentConnectionFieldType',
  'selectedNodes',
  'selectedEdges',
  'isReady',
  'nodesToCopy',
  'edgesToCopy',
  'connectionMade',
  'modifyingEdge',
  'addNewNodePosition',
];