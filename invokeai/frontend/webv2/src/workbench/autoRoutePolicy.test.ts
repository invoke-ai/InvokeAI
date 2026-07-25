import type { InvocationRoute } from '@workbench/invocationContracts';

import { describe, expect, it } from 'vitest';

import type { CanvasProjectMutation } from './canvasProjectMutations';

import {
  autoSwitchDestinations,
  getChangedValueKeys,
  getRouteAfterHighConfidenceEdit,
  isHighConfidenceCanvasEdit,
  isHighConfidenceCanvasEditIntent,
  isHighConfidenceGenerateEdit,
  isHighConfidenceGraphEdit,
  isHighConfidenceUpscaleEdit,
} from './autoRoutePolicy';

const mutation = (value: Record<string, unknown>): CanvasProjectMutation => value as never;

const route = (overrides: Partial<InvocationRoute> = {}): InvocationRoute => ({
  destination: 'canvas',
  destinationLocked: false,
  sourceId: 'generate',
  sourceLocked: false,
  ...overrides,
});

describe('getRouteAfterHighConfidenceEdit', () => {
  it('maps each new source to its default destination', () => {
    expect(getRouteAfterHighConfidenceEdit(route({ sourceId: 'canvas' }), 'generate')).toEqual(
      route({ destination: 'gallery', sourceId: 'generate' })
    );
    expect(getRouteAfterHighConfidenceEdit(route(), 'workflow')).toEqual(
      route({ destination: 'gallery', sourceId: 'workflow' })
    );
    expect(getRouteAfterHighConfidenceEdit(route(), 'upscale')).toEqual(
      route({ destination: 'gallery', sourceId: 'upscale' })
    );
    expect(getRouteAfterHighConfidenceEdit(route({ destination: 'gallery' }), 'canvas')).toEqual(
      route({ destination: 'canvas', sourceId: 'canvas' })
    );
    expect(Object.keys(autoSwitchDestinations).sort()).toEqual(['canvas', 'generate', 'upscale', 'workflow']);
  });

  it('preserves route identity for same-source and source-locked edits', () => {
    const sameSource = route({ destination: 'gallery' });
    const locked = route({ sourceLocked: true });

    expect(getRouteAfterHighConfidenceEdit(sameSource, 'generate')).toBe(sameSource);
    expect(getRouteAfterHighConfidenceEdit(locked, 'canvas')).toBe(locked);
  });

  it('preserves a locked destination while changing source', () => {
    expect(getRouteAfterHighConfidenceEdit(route({ destinationLocked: true }), 'workflow')).toEqual(
      route({ destinationLocked: true, sourceId: 'workflow' })
    );
  });
});

describe('widget edit classification', () => {
  it('detects changed keys with Object.is semantics', () => {
    expect(
      getChangedValueKeys({ height: 512, model: null, seed: Number.NaN }, { height: 512, seed: Number.NaN })
    ).toEqual([]);
    expect(getChangedValueKeys({ height: 512 }, { height: 768, unknown: true })).toEqual(['height', 'unknown']);
  });

  it('ignores only documented Generate and Upscale UI noise', () => {
    expect(
      isHighConfidenceGenerateEdit([
        'aspectRatioIsLocked',
        'batchCount',
        'negativePromptHeightPx',
        'positivePromptHeightPx',
      ])
    ).toBe(false);
    expect(isHighConfidenceUpscaleEdit(['batchCount', 'negativePromptHeightPx', 'positivePromptHeightPx'])).toBe(false);
    expect(isHighConfidenceGenerateEdit(['steps'])).toBe(true);
    expect(isHighConfidenceUpscaleEdit(['creativity'])).toBe(true);
    expect(isHighConfidenceGenerateEdit(['futureSetting'])).toBe(true);
    expect(isHighConfidenceUpscaleEdit(['futureSetting'])).toBe(true);
  });
});

describe('Canvas edit classification', () => {
  it('counts structural layer-stack and visibility changes but not selection-only changes', () => {
    expect(
      isHighConfidenceCanvasEdit(
        mutation({
          add: { index: 0, layers: [{}] },
          enabledUpdates: [],
          selectedLayerId: 'new-layer',
          type: 'applyCanvasLayerStackMutation',
        })
      )
    ).toBe(true);
    expect(
      isHighConfidenceCanvasEdit(
        mutation({
          enabledUpdates: [],
          removeIds: ['old-layer'],
          selectedLayerId: null,
          type: 'applyCanvasLayerStackMutation',
        })
      )
    ).toBe(true);
    expect(
      isHighConfidenceCanvasEdit(
        mutation({
          enabledUpdates: [{ id: 'layer', isEnabled: false }],
          selectedLayerId: 'layer',
          type: 'applyCanvasLayerStackMutation',
        })
      )
    ).toBe(true);
    expect(
      isHighConfidenceCanvasEdit(
        mutation({ enabledUpdates: [], selectedLayerId: 'layer', type: 'applyCanvasLayerStackMutation' })
      )
    ).toBe(false);
    expect(isHighConfidenceCanvasEdit(mutation({ type: 'setCanvasLayersEnabled', updates: [] }))).toBe(false);
    expect(
      isHighConfidenceCanvasEdit(
        mutation({ type: 'setCanvasLayersEnabled', updates: [{ id: 'layer', isEnabled: false }] })
      )
    ).toBe(true);
  });

  it('counts composite layer patches but ignores metadata-only patches', () => {
    for (const patch of [{ blendMode: 'multiply' }, { isEnabled: false }, { opacity: 0.5 }, { transform: { x: 1 } }]) {
      expect(isHighConfidenceCanvasEdit(mutation({ patch, type: 'updateCanvasLayer' }))).toBe(true);
    }
    expect(
      isHighConfidenceCanvasEdit(mutation({ patch: { isLocked: true, name: 'Layer 1' }, type: 'updateCanvasLayer' }))
    ).toBe(false);
  });

  it('treats committed paint as intent and delegates mutation intents', () => {
    expect(isHighConfidenceCanvasEditIntent({ kind: 'paint' })).toBe(true);
    expect(
      isHighConfidenceCanvasEditIntent({
        kind: 'mutation',
        mutation: mutation({ id: 'layer', type: 'setCanvasSelectedLayer' }),
      })
    ).toBe(false);
  });
});

describe('Workflow edit classification', () => {
  it('counts graph topology and field values while ignoring presentation changes', () => {
    for (const type of [
      'addEdge',
      'addGraphElements',
      'addNode',
      'addNodeAndEdge',
      'removeEdges',
      'removeNodes',
      'setFieldValue',
    ] as const) {
      expect(isHighConfidenceGraphEdit({ type } as never)).toBe(true);
    }
    for (const type of ['setMetadata', 'setNodeLabel', 'setNodePosition'] as const) {
      expect(isHighConfidenceGraphEdit({ type } as never)).toBe(false);
    }
  });
});
