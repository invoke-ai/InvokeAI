import type {
  CanvasDocumentContractV3,
  CanvasGroupContract,
  CanvasLayerContract,
  CanvasNodeContract,
  CanvasStackForests,
} from '@workbench/canvas-engine/contracts';

import { createEmptyStacks } from '@workbench/canvas-engine/document/documentTree';
import { LAYER_STACK_ORDER, layerStackOf } from '@workbench/canvas-engine/document/layerStacks';

const base = (id: string) => ({
  blendMode: 'normal' as const,
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
});

const mask = () => ({ bitmap: null, fill: { color: '#e07575', style: 'diagonal' as const } });

export const layerContract = (
  id: string,
  type: CanvasLayerContract['type'] = 'raster',
  overrides: Partial<CanvasLayerContract> = {}
): CanvasLayerContract => {
  const contract = ((): CanvasLayerContract => {
    switch (type) {
      case 'raster':
        return { ...base(id), source: { bitmap: null, type: 'paint' }, type };
      case 'control':
        return {
          ...base(id),
          adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
          source: { bitmap: null, type: 'paint' },
          type,
          withTransparencyEffect: false,
        };
      case 'inpaint_mask':
        return { ...base(id), mask: mask(), type };
      case 'regional_guidance':
        return {
          ...base(id),
          autoNegative: false,
          mask: mask(),
          negativePrompt: null,
          positivePrompt: null,
          referenceImages: [],
          type,
        };
    }
  })();
  return { ...contract, ...overrides } as CanvasLayerContract;
};

export const groupContract = (
  id: string,
  children: readonly CanvasNodeContract[] = [],
  overrides: Partial<Omit<CanvasGroupContract, 'type' | 'children'>> = {}
): CanvasGroupContract => ({
  children: [...children],
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  type: 'group',
  ...overrides,
});

const stackOfNode = (node: CanvasNodeContract): CanvasLayerContract['type'] | null => {
  if (node.type !== 'group') {
    return layerStackOf(node);
  }
  for (const child of node.children) {
    const stack = stackOfNode(child);
    if (stack) {
      return stack;
    }
  }
  return null;
};

export const stacksFrom = (nodes: readonly CanvasNodeContract[]): CanvasStackForests => {
  const stacks = createEmptyStacks();
  for (const node of nodes) {
    stacks[stackOfNode(node) ?? 'raster'].push(node);
  }
  return stacks;
};

export const documentFrom = (
  stacks: Partial<CanvasStackForests> | readonly CanvasNodeContract[],
  selectedLayerId: string | null = null
): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 512, width: 512, x: 0, y: 0 },
  height: 512,
  selectedLayerId,
  stacks: Array.isArray(stacks)
    ? stacksFrom(stacks)
    : { ...createEmptyStacks(), ...(stacks as Partial<CanvasStackForests>) },
  version: 3,
  width: 512,
});

export const flatDocument = documentFrom;

export const createLargeFlatDocument = (
  count: number,
  selectedLayerId: string | null = 'l0'
): CanvasDocumentContractV3 =>
  documentFrom(
    Array.from({ length: count }, (_, index) =>
      layerContract(`l${index}`, LAYER_STACK_ORDER[index % LAYER_STACK_ORDER.length]!)
    ),
    selectedLayerId
  );

export const createLargeTreeDocument = (
  count: number,
  fanout = 8,
  selectedLayerId: string | null = 'l0'
): CanvasDocumentContractV3 => {
  const stacks = createEmptyStacks();
  let created = 0;
  let leaves = 0;
  let groups = 0;
  const build = (stack: CanvasLayerContract['type'], budget: number, depth: number): CanvasNodeContract[] => {
    const nodes: CanvasNodeContract[] = [];
    while (budget > 0 && created < count) {
      if (depth < 3 && budget > fanout) {
        const before = created;
        const id = `g${groups++}`;
        created += 1;
        const children = build(stack, Math.min(budget - 1, fanout * fanout), depth + 1);
        nodes.push(groupContract(id, children));
        budget -= created - before;
      } else {
        nodes.push(layerContract(`l${leaves++}`, stack));
        created += 1;
        budget -= 1;
      }
    }
    return nodes;
  };
  const perStack = Math.ceil(count / LAYER_STACK_ORDER.length);
  for (const stack of LAYER_STACK_ORDER) {
    stacks[stack].push(...build(stack, perStack, 0));
  }
  return documentFrom(stacks, selectedLayerId);
};
