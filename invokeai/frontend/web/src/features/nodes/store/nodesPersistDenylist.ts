import { NodesState } from './nodesSlice';

/**
 * Nodes slice persist denylist
 */
const itemsToDenylist: (keyof NodesState)[] = ['schema', 'invocationTemplates'];

export const nodesDenylist = itemsToDenylist.map(
  (denylistItem) => `nodes.${denylistItem}`
);
