import { getInitialCanvasState } from 'features/controlLayers/store/types';
import { getVectorLayerState } from 'features/controlLayers/store/util';
import { describe, expect, it } from 'vitest';

import { parseCanvasProjectState } from './canvasProjectFile';

describe('parseCanvasProjectState', () => {
  const project = (vectorLayers: unknown[]) => ({
    rasterLayers: [],
    controlLayers: [],
    inpaintMasks: [],
    regionalGuidance: [],
    vectorLayers,
    bbox: getInitialCanvasState().bbox,
    selectedEntityIdentifier: null,
    bookmarkedEntityIdentifier: null,
  });
  it('accepts valid empty vector layers', () => {
    const layer = getVectorLayerState('layer');
    expect(parseCanvasProjectState(project([layer])).vectorLayers).toEqual([layer]);
  });
  it.each([
    { id: 'layer', type: 'vector_layer' },
    { ...getVectorLayerState('layer'), paths: null },
    {
      ...getVectorLayerState('layer'),
      paths: [{ id: 'p', name: null, isClosed: false, points: [{ anchor: { x: 'bad', y: 0 } }] }],
    },
  ])('rejects malformed vector geometry', (layer) => {
    expect(() => parseCanvasProjectState(project([layer]))).toThrow();
  });
  it('adds default vector layers when loading a legacy project file', () => {
    const initialState = getInitialCanvasState();
    const legacyProjectState = {
      rasterLayers: [],
      controlLayers: [],
      inpaintMasks: [],
      regionalGuidance: [],
      bbox: initialState.bbox,
      selectedEntityIdentifier: null,
      bookmarkedEntityIdentifier: null,
    };

    const parsed = parseCanvasProjectState(legacyProjectState);

    expect(parsed.vectorLayers).toEqual([]);
  });
});
