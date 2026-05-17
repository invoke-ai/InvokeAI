import { describe, expect, it } from 'vitest';

import { getInitialCanvasState } from 'features/controlLayers/store/types';

import { parseCanvasProjectState } from './canvasProjectFile';

describe('parseCanvasProjectState', () => {
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
