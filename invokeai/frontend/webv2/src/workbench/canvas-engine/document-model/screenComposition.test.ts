import type { CanvasLayerContract, CanvasNodeContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import type { CanvasScreenViewState } from './screenComposition';

import { documentFrom, groupContract, layerContract } from './documentFixtures.testStub';
import { compileDocumentLeaves } from './documentModel';
import { planScreenComposition } from './screenComposition';

const layer = (id: string, type: CanvasLayerContract['type'], overrides: Partial<CanvasLayerContract> = {}) =>
  layerContract(id, type, overrides);

const leaves = (nodes: readonly CanvasNodeContract[]) => compileDocumentLeaves(documentFrom(nodes));

const view = (overrides: Partial<CanvasScreenViewState> = {}): CanvasScreenViewState => ({
  isolationLayerId: null,
  showOverlayStacks: { control: true, inpaint_mask: true, regional_guidance: true },
  ...overrides,
});

const drawn = (plan: ReturnType<typeof planScreenComposition>): string[] => plan.leaves.map((leaf) => leaf.id);

describe('planScreenComposition', () => {
  it('draws stacks bottom first and each stack bottom-most leaf first', () => {
    const plan = planScreenComposition(
      leaves([layer('m1', 'inpaint_mask'), layer('c1', 'control'), layer('r1', 'raster'), layer('r2', 'raster')]),
      view()
    );
    expect(drawn(plan)).toEqual(['r2', 'r1', 'c1', 'm1']);
  });

  it('skips disabled and document-hidden leaves and leaves an ancestor gates its descendants', () => {
    const plan = planScreenComposition(
      leaves([
        layer('r1', 'raster', { isEnabled: false }),
        groupContract('off', [layer('r2', 'raster')], { isEnabled: false }),
        groupContract('hidden', [layer('c2', 'control')], { isHidden: true }),
        layer('c1', 'control', { isHidden: true } as Partial<CanvasLayerContract>),
        layer('r3', 'raster'),
      ]),
      view()
    );
    expect(drawn(plan)).toEqual(['r3']);
  });

  it('switches whole overlay stacks off without touching raster', () => {
    const plan = planScreenComposition(
      leaves([layer('m1', 'inpaint_mask'), layer('c1', 'control'), layer('r1', 'raster')]),
      view({ showOverlayStacks: { control: false, inpaint_mask: true, regional_guidance: true } })
    );
    expect(drawn(plan)).toEqual(['r1', 'm1']);
  });

  it('isolates one leaf, overriding its own and its ancestors visibility and every stack switch', () => {
    const plan = planScreenComposition(
      leaves([
        groupContract(
          'g',
          [layer('c1', 'control', { isEnabled: false, isHidden: true } as Partial<CanvasLayerContract>)],
          {
            isEnabled: false,
          }
        ),
        layer('r1', 'raster'),
      ]),
      view({
        isolationLayerId: 'c1',
        showOverlayStacks: { control: false, inpaint_mask: false, regional_guidance: false },
      })
    );
    expect(drawn(plan)).toEqual(['c1']);
    expect(plan.isolationLayerId).toBe('c1');
  });

  it('isolates a group as every leaf beneath it', () => {
    const plan = planScreenComposition(
      leaves([
        groupContract('g', [
          layer('r1', 'raster'),
          groupContract('inner', [layer('r2', 'raster', { isEnabled: false })]),
        ]),
        layer('r3', 'raster'),
      ]),
      view({ isolationLayerId: 'g' })
    );
    expect(drawn(plan)).toEqual(['r2', 'r1']);
  });
});
