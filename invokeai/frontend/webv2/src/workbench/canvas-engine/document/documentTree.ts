import type {
  CanvasGroupContract,
  CanvasLayerContract,
  CanvasLayerStackKind,
  CanvasNodeContract,
  CanvasStackForests,
} from '@workbench/canvas-engine/contracts';

export const isGroupNode = (node: CanvasNodeContract): node is CanvasGroupContract => node.type === 'group';

export const isLeafNode = (node: CanvasNodeContract): node is CanvasLayerContract => node.type !== 'group';

/** The child list `parentId` names in `stack`: the forest roots for `null`, else the group's children. */
export const childrenOf = (
  stacks: CanvasStackForests,
  stack: CanvasLayerStackKind,
  parentId: string | null
): readonly CanvasNodeContract[] | null => {
  if (parentId === null) {
    return stacks[stack];
  }
  const group = findNode(stacks[stack], parentId);
  return group && isGroupNode(group) ? group.children : null;
};

const findNode = (nodes: readonly CanvasNodeContract[], id: string): CanvasNodeContract | null => {
  for (const node of nodes) {
    if (node.id === id) {
      return node;
    }
    if (isGroupNode(node)) {
      const found = findNode(node.children, id);
      if (found) {
        return found;
      }
    }
  }
  return null;
};

/** Every node of the subtree rooted at `node`, preorder, including `node`. */
export const collectSubtree = (node: CanvasNodeContract, into: CanvasNodeContract[] = []): CanvasNodeContract[] => {
  into.push(node);
  if (isGroupNode(node)) {
    for (const child of node.children) {
      collectSubtree(child, into);
    }
  }
  return into;
};

/** The leaves of the subtree rooted at `node`, preorder. */
export const collectSubtreeLeaves = (
  node: CanvasNodeContract,
  into: CanvasLayerContract[] = []
): CanvasLayerContract[] => {
  if (isGroupNode(node)) {
    for (const child of node.children) {
      collectSubtreeLeaves(child, into);
    }
  } else {
    into.push(node);
  }
  return into;
};

/** How many levels of groups `node` nests: 0 for a leaf or an empty group. */
export const subtreeDepth = (node: CanvasNodeContract): number =>
  isGroupNode(node) ? node.children.reduce((deepest, child) => Math.max(deepest, subtreeDepth(child) + 1), 0) : 0;

/** A deep copy of the subtree with every id re-minted; `ids` maps old ids to new ones. */
export const cloneSubtree = (
  node: CanvasNodeContract,
  createId: () => string,
  ids = new Map<string, string>()
): { node: CanvasNodeContract; ids: Map<string, string> } => {
  const id = createId();
  ids.set(node.id, id);
  if (isGroupNode(node)) {
    const children = node.children.map((child) => cloneSubtree(child, createId, ids).node);
    return { ids, node: { ...node, children, id } };
  }
  return { ids, node: { ...structuredClone(node), id } };
};

const replaceIn = (
  nodes: readonly CanvasNodeContract[],
  parentId: string,
  replacement: (children: readonly CanvasNodeContract[]) => readonly CanvasNodeContract[]
): readonly CanvasNodeContract[] => {
  let changed = false;
  const next = nodes.map((node) => {
    if (!isGroupNode(node)) {
      return node;
    }
    const children =
      node.id === parentId ? replacement(node.children) : replaceIn(node.children, parentId, replacement);
    if (children === node.children) {
      return node;
    }
    changed = true;
    return { ...node, children: [...children] };
  });
  return changed ? next : nodes;
};

/** Replaces one child list, sharing every untouched node; the same forests when nothing changed. */
export const replaceChildren = (
  stacks: CanvasStackForests,
  stack: CanvasLayerStackKind,
  parentId: string | null,
  children: readonly CanvasNodeContract[]
): CanvasStackForests => {
  const roots = parentId === null ? children : replaceIn(stacks[stack], parentId, () => children);
  return roots === stacks[stack] ? stacks : { ...stacks, [stack]: [...roots] };
};

/** Removes the subtrees rooted at `ids`, sharing every untouched node. */
export const removeNodes = (stacks: CanvasStackForests, ids: ReadonlySet<string>): CanvasStackForests => {
  if (ids.size === 0) {
    return stacks;
  }
  const visit = (nodes: readonly CanvasNodeContract[]): readonly CanvasNodeContract[] => {
    let changed = false;
    const next: CanvasNodeContract[] = [];
    for (const node of nodes) {
      if (ids.has(node.id)) {
        changed = true;
        continue;
      }
      if (isGroupNode(node)) {
        const children = visit(node.children);
        if (children !== node.children) {
          changed = true;
          next.push({ ...node, children: [...children] });
          continue;
        }
      }
      next.push(node);
    }
    return changed ? next : nodes;
  };
  let result = stacks;
  for (const stack of Object.keys(stacks) as CanvasLayerStackKind[]) {
    const roots = visit(stacks[stack]);
    if (roots !== stacks[stack]) {
      result = result === stacks ? { ...stacks } : result;
      result[stack] = [...roots];
    }
  }
  return result;
};

/** Inserts `nodes` into the child list `parentId` names at `index` (clamped), sharing every other node. */
export const insertNodes = (
  stacks: CanvasStackForests,
  stack: CanvasLayerStackKind,
  parentId: string | null,
  index: number,
  nodes: readonly CanvasNodeContract[]
): CanvasStackForests | null => {
  const children = childrenOf(stacks, stack, parentId);
  if (!children) {
    return null;
  }
  const at = Math.max(0, Math.min(index, children.length));
  return replaceChildren(stacks, stack, parentId, [...children.slice(0, at), ...nodes, ...children.slice(at)]);
};

export const EMPTY_STACKS: CanvasStackForests = {
  control: [],
  inpaint_mask: [],
  raster: [],
  regional_guidance: [],
};

export const createEmptyStacks = (): CanvasStackForests => ({
  control: [],
  inpaint_mask: [],
  raster: [],
  regional_guidance: [],
});
