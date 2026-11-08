import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';

import {
  documentFrom,
  groupContract,
  layerContract,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import type { GroupCompositeFacts } from './groupCompositeScopes';

import { collectCompositedGroups, planGroupCompositeScopes } from './groupCompositeScopes';

const stack = (id: string): CanvasAdjustmentsContract => [
  { brightness: 0.2, contrast: 0, id, isEnabled: true, type: 'brightness-contrast' },
];
const identityStack: CanvasAdjustmentsContract = [
  { brightness: 0, contrast: 0, id: 'noop', isEnabled: true, type: 'brightness-contrast' },
];
const facts = (
  adjustments: CanvasAdjustmentsContract,
  overrides: Partial<GroupCompositeFacts> = {}
): GroupCompositeFacts => ({
  adjustments,
  blendMode: 'normal',
  opacity: 1,
  ...overrides,
});

describe('collectCompositedGroups', () => {
  it('collects only raster-stack groups with contributing stacks', () => {
    const doc = documentFrom({
      control: [groupContract('og', [layerContract('c1', 'control')], { adjustments: stack('a') })],
      raster: [
        groupContract('g1', [layerContract('r1')], { adjustments: stack('b') }),
        groupContract('g2', [layerContract('r2')], { adjustments: identityStack }),
        groupContract('g3', [layerContract('r3')]),
      ],
    });
    expect([...collectCompositedGroups(doc).keys()]).toEqual(['g1']);
  });

  it('collects raster-stack groups scoped only by opacity or blend, with an empty stack', () => {
    const doc = documentFrom({
      raster: [
        groupContract('faded', [layerContract('r1')], { opacity: 0.5 }),
        groupContract('mult', [layerContract('r2')], { blendMode: 'multiply' }),
        groupContract('plain', [layerContract('r3')], { blendMode: 'normal', opacity: 1 }),
        groupContract('identity', [layerContract('r4')], { adjustments: identityStack, opacity: 0.25 }),
      ],
    });
    const composited = collectCompositedGroups(doc);
    expect([...composited.keys()]).toEqual(['faded', 'mult', 'identity']);
    expect(composited.get('faded')).toEqual(facts([], { opacity: 0.5 }));
    expect(composited.get('mult')).toEqual(facts([], { blendMode: 'multiply' }));
    // An identity stack contributes nothing: the scope keeps an EMPTY stack so
    // renderers can skip the pixel pass, but the opacity still isolates it.
    expect(composited.get('identity')).toEqual(facts([], { opacity: 0.25 }));
  });
});

describe('planGroupCompositeScopes', () => {
  const adjusted = new Map<string, GroupCompositeFacts>([
    ['g', facts(stack('g'))],
    ['h', facts(stack('h'), { blendMode: 'multiply', opacity: 0.5 })],
  ]);
  const scopeG = { adjustments: stack('g'), blendMode: 'normal', opacity: 1 } as const;
  const scopeH = { adjustments: stack('h'), blendMode: 'multiply', opacity: 0.5 } as const;

  it('returns no scopes when nothing is composited', () => {
    expect(planGroupCompositeScopes([{ parentIds: [] }, { parentIds: ['x'] }], new Map())).toEqual([]);
  });

  it('covers a contiguous run and ignores non-composited ancestors', () => {
    const scopes = planGroupCompositeScopes(
      [{ parentIds: [] }, { parentIds: ['plain', 'g'] }, { parentIds: ['plain', 'g'] }, { parentIds: [] }],
      adjusted
    );
    expect(scopes).toEqual([{ ...scopeG, children: [], end: 3, id: 'g', start: 1 }]);
  });

  it('nests an inner composited group inside its outer scope, carrying its opacity and blend', () => {
    const scopes = planGroupCompositeScopes(
      [{ parentIds: ['g'] }, { parentIds: ['g', 'h'] }, { parentIds: ['g', 'h'] }, { parentIds: ['g'] }],
      adjusted
    );
    expect(scopes).toEqual([
      {
        ...scopeG,
        children: [{ ...scopeH, children: [], end: 3, id: 'h', start: 1 }],
        end: 4,
        id: 'g',
        start: 0,
      },
    ]);
  });

  it('closes a scope that runs to the end of the list and separates sibling scopes', () => {
    const scopes = planGroupCompositeScopes(
      [{ parentIds: ['g'] }, { parentIds: [] }, { parentIds: ['h'] }, { parentIds: ['h'] }],
      adjusted
    );
    expect(scopes).toEqual([
      { ...scopeG, children: [], end: 1, id: 'g', start: 0 },
      { ...scopeH, children: [], end: 4, id: 'h', start: 2 },
    ]);
  });

  it('works identically on a reversed (bottom-first) list, as subtrees stay contiguous', () => {
    const items = [{ parentIds: [] }, { parentIds: ['g'] }, { parentIds: ['g', 'h'] }, { parentIds: ['g'] }];
    const scopes = planGroupCompositeScopes([...items].reverse(), adjusted);
    expect(scopes).toEqual([
      {
        ...scopeG,
        children: [{ ...scopeH, children: [], end: 2, id: 'h', start: 1 }],
        end: 3,
        id: 'g',
        start: 0,
      },
    ]);
  });
});
