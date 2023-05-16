import { NodesState } from './nodesSlice';

/**
 * Nodes slice persist denylist
 */
export const nodesPersistDenylist: (keyof NodesState)[] = [
  'schema',
  'invocationTemplates',
];
