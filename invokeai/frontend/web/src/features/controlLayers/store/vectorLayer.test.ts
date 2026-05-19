import type { RootState } from 'app/store/store';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { getPrefixedIdMock } = vi.hoisted(() => ({
  getPrefixedIdMock: vi.fn(),
}));

vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: getPrefixedIdMock,
}));

import {
  canvasProjectRecalled,
  canvasSliceConfig,
  entityDuplicated,
  entityReset,
  vectorLayerAdded,
  vectorLayerPathsReplaced,
  vectorPathAdded,
} from './canvasSlice';
import { buildSelectHasObjects, selectAllEntities, selectCanvasMetadata } from './selectors';
import { getInitialCanvasState, zCanvasMetadata, zCanvasState } from './types';
import { getVectorLayerState } from './util';

describe('vector layer integration', () => {
  const { reducer } = canvasSliceConfig.slice;

  beforeEach(() => {
    let id = 0;
    getPrefixedIdMock.mockReset();
    getPrefixedIdMock.mockImplementation((prefix: string) => `${prefix}-${++id}`);
  });

  it('migrates legacy persisted canvas state without vector layers', () => {
    const initialState = getInitialCanvasState();
    const { vectorLayers: _vectorLayers, ...legacyState } = initialState;

    const parsed = zCanvasState.parse(legacyState);

    expect(parsed.vectorLayers).toEqual({ isHidden: false, entities: [] });
  });

  it('migrates legacy canvas metadata without vector layers', () => {
    const parsed = zCanvasMetadata.parse({
      rasterLayers: [],
      controlLayers: [],
      inpaintMasks: [],
      regionalGuidance: [],
    });

    expect(parsed.vectorLayers).toEqual([]);
  });

  it('adds a vector layer with empty paths and selects it', () => {
    const state = reducer(getInitialCanvasState(), vectorLayerAdded({ isSelected: true }));

    expect(state.vectorLayers.entities).toHaveLength(1);
    expect(state.vectorLayers.entities[0]).toMatchObject({
      id: 'vector_layer-1',
      type: 'vector_layer',
      paths: [],
      opacity: 1,
      position: { x: 0, y: 0 },
    });
    expect(state.selectedEntityIdentifier).toEqual({ id: 'vector_layer-1', type: 'vector_layer' });
  });

  it('adds a bezier path to an existing vector layer', () => {
    const state = getInitialCanvasState();
    state.vectorLayers.entities.push(getVectorLayerState('vector-layer-a'));

    const result = reducer(
      state,
      vectorPathAdded({
        entityIdentifier: { id: 'vector-layer-a', type: 'vector_layer' },
        path: {
          id: 'bezier-path-a',
          name: 'Path A',
          isClosed: false,
          points: [
            { anchor: { x: 10, y: 20 }, inHandle: null, outHandle: null, type: 'corner' },
            { anchor: { x: 30, y: 40 }, inHandle: null, outHandle: null, type: 'corner' },
          ],
        },
      })
    );

    expect(result.vectorLayers.entities[0]?.paths).toHaveLength(1);
    expect(result.vectorLayers.entities[0]?.paths[0]).toMatchObject({
      id: 'bezier-path-a',
      isClosed: false,
    });
  });

  it('replaces all bezier paths on an existing vector layer', () => {
    const state = getInitialCanvasState();
    state.vectorLayers.entities.push(
      getVectorLayerState('vector-layer-a', {
        paths: [
          {
            id: 'bezier-path-a',
            name: 'Old Path',
            isClosed: false,
            points: [
              { anchor: { x: 10, y: 20 }, inHandle: null, outHandle: null, type: 'corner' },
              { anchor: { x: 30, y: 40 }, inHandle: null, outHandle: null, type: 'corner' },
            ],
          },
        ],
      })
    );

    const result = reducer(
      state,
      vectorLayerPathsReplaced({
        entityIdentifier: { id: 'vector-layer-a', type: 'vector_layer' },
        paths: [
          {
            id: 'bezier-path-b',
            name: 'New Path',
            isClosed: false,
            points: [
              { anchor: { x: 1, y: 2 }, inHandle: null, outHandle: null, type: 'corner' },
              { anchor: { x: 3, y: 4 }, inHandle: null, outHandle: null, type: 'corner' },
            ],
          },
        ],
      })
    );

    expect(result.vectorLayers.entities[0]?.paths).toEqual([
      {
        id: 'bezier-path-b',
        name: 'New Path',
        isClosed: false,
        points: [
          { anchor: { x: 1, y: 2 }, inHandle: null, outHandle: null, type: 'corner' },
          { anchor: { x: 3, y: 4 }, inHandle: null, outHandle: null, type: 'corner' },
        ],
      },
    ]);
  });

  it('duplicates a vector layer and rekeys its paths', () => {
    const vectorLayer = getVectorLayerState('vector-layer-a', {
      name: 'Spline Layer',
      paths: [
        {
          id: 'bezier-path-a',
          name: 'Path A',
          isClosed: false,
          points: [
            {
              anchor: { x: 10, y: 20 },
              inHandle: null,
              outHandle: { x: 18, y: 28 },
              type: 'smooth',
            },
          ],
        },
      ],
    });
    const state = getInitialCanvasState();
    state.vectorLayers.entities.push(vectorLayer);

    const result = reducer(
      state,
      entityDuplicated({ entityIdentifier: { id: 'vector-layer-a', type: 'vector_layer' } })
    );

    expect(result.vectorLayers.entities).toHaveLength(2);
    expect(result.vectorLayers.entities[1]).toMatchObject({
      id: 'vector_layer-1',
      name: 'Spline Layer (Copy)',
      type: 'vector_layer',
    });
    expect(result.vectorLayers.entities[1]?.paths).toHaveLength(1);
    expect(result.vectorLayers.entities[1]?.paths[0]?.id).toBe('bezier_path-2');
    expect(result.vectorLayers.entities[1]?.paths[0]?.id).not.toBe('bezier-path-a');
    expect(result.selectedEntityIdentifier).toEqual({ id: 'vector_layer-1', type: 'vector_layer' });
  });

  it('resets vector layer paths and position', () => {
    const state = getInitialCanvasState();
    state.vectorLayers.entities.push(
      getVectorLayerState('vector-layer-a', {
        position: { x: 64, y: 32 },
        isEnabled: false,
        paths: [
          {
            id: 'bezier-path-a',
            name: null,
            isClosed: false,
            points: [{ anchor: { x: 1, y: 2 }, inHandle: null, outHandle: null, type: 'corner' }],
          },
        ],
      })
    );

    const result = reducer(state, entityReset({ entityIdentifier: { id: 'vector-layer-a', type: 'vector_layer' } }));

    expect(result.vectorLayers.entities[0]).toMatchObject({
      isEnabled: true,
      position: { x: 0, y: 0 },
      paths: [],
    });
  });

  it('includes vector layers in entity ordering, content selectors, metadata, and project recall', () => {
    const state = getInitialCanvasState();
    state.rasterLayers.entities.push({
      id: 'raster-a',
      name: null,
      type: 'raster_layer',
      isEnabled: true,
      isLocked: false,
      objects: [],
      opacity: 1,
      position: { x: 0, y: 0 },
    });
    state.controlLayers.entities.push({
      id: 'control-a',
      name: null,
      type: 'control_layer',
      isEnabled: true,
      isLocked: false,
      withTransparencyEffect: true,
      objects: [],
      opacity: 1,
      position: { x: 0, y: 0 },
      controlAdapter: {
        type: 'controlnet',
        model: null,
        weight: 0.75,
        beginEndStepPct: [0, 0.75],
        controlMode: 'balanced',
      },
    });
    state.vectorLayers.entities.push(
      getVectorLayerState('vector-a', {
        paths: [
          {
            id: 'bezier-path-a',
            name: null,
            isClosed: false,
            points: [{ anchor: { x: 5, y: 6 }, inHandle: null, outHandle: null, type: 'corner' }],
          },
        ],
      })
    );
    state.regionalGuidance.entities.push({
      id: 'regional-a',
      name: null,
      type: 'regional_guidance',
      isEnabled: true,
      isLocked: false,
      objects: [],
      fill: { style: 'solid', color: { r: 1, g: 2, b: 3 } },
      opacity: 0.5,
      position: { x: 0, y: 0 },
      autoNegative: false,
      positivePrompt: null,
      negativePrompt: null,
      referenceImages: [],
    });
    state.inpaintMasks.entities.push({
      id: 'mask-a',
      name: null,
      type: 'inpaint_mask',
      isEnabled: true,
      isLocked: false,
      objects: [],
      fill: { style: 'diagonal', color: { r: 4, g: 5, b: 6 } },
      opacity: 1,
      position: { x: 0, y: 0 },
    });

    expect(selectAllEntities(state).map((entity) => entity.type)).toEqual([
      'inpaint_mask',
      'regional_guidance',
      'vector_layer',
      'control_layer',
      'raster_layer',
    ]);

    const rootState = { canvas: { present: state } } as RootState;
    expect(buildSelectHasObjects({ id: 'vector-a', type: 'vector_layer' })(rootState)).toBe(true);
    expect(selectCanvasMetadata(rootState).canvas_v2_metadata.vectorLayers).toEqual(state.vectorLayers.entities);

    const recalled = reducer(
      getInitialCanvasState(),
      canvasProjectRecalled({
        rasterLayers: state.rasterLayers.entities,
        controlLayers: state.controlLayers.entities,
        vectorLayers: state.vectorLayers.entities,
        inpaintMasks: state.inpaintMasks.entities,
        regionalGuidance: state.regionalGuidance.entities,
        bbox: state.bbox,
        selectedEntityIdentifier: { id: 'vector-a', type: 'vector_layer' },
        bookmarkedEntityIdentifier: null,
      })
    );

    expect(recalled.vectorLayers.entities).toEqual(state.vectorLayers.entities);
    expect(recalled.selectedEntityIdentifier).toEqual({ id: 'vector-a', type: 'vector_layer' });
  });
});
