import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { z } from 'zod';

const { getPrefixedIdMock } = vi.hoisted(() => ({
  getPrefixedIdMock: vi.fn((prefix: string) => `${prefix}-generated`),
}));

vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: getPrefixedIdMock,
}));

import {
  canvasSessionReset,
  canvasSessionSliceConfig,
  canvasSessionThumbnailsVisibilityToggled,
} from './canvasStagingAreaSlice';

describe('canvasStagingAreaSlice', () => {
  type InitialState = ReturnType<typeof canvasSessionSliceConfig.getInitialState>;
  type SchemaState = z.infer<typeof canvasSessionSliceConfig.schema>;

  const { reducer } = canvasSessionSliceConfig.slice;
  const migrate = canvasSessionSliceConfig.persistConfig?.migrate;

  beforeEach(() => {
    getPrefixedIdMock.mockReset();
    getPrefixedIdMock.mockImplementation((prefix: string) => `${prefix}-generated`);
  });

  it('keeps the initial state aligned with the persisted schema', () => {
    assert<Equals<InitialState, SchemaState>>();
  });

  it('toggles thumbnail visibility', () => {
    const state = canvasSessionSliceConfig.getInitialState();

    const hidden = reducer(state, canvasSessionThumbnailsVisibilityToggled());
    const shown = reducer(hidden, canvasSessionThumbnailsVisibilityToggled());

    expect(hidden.areThumbnailsVisible).toBe(false);
    expect(shown.areThumbnailsVisible).toBe(true);
  });

  it('resets thumbnails visibility and discarded items on session reset', () => {
    const state = {
      _version: 2 as const,
      canvasSessionId: 'canvas-existing',
      canvasDiscardedQueueItems: [1, 2],
      areThumbnailsVisible: false,
    };

    getPrefixedIdMock.mockReturnValueOnce('canvas-reset');

    const result = reducer(state, canvasSessionReset());

    expect(result).toEqual({
      _version: 2,
      canvasSessionId: 'canvas-reset',
      canvasDiscardedQueueItems: [],
      areThumbnailsVisible: true,
    });
  });

  it('migrates legacy persisted state without a version to v2 defaults', () => {
    expect(migrate).toBeDefined();

    const result = migrate?.({});

    expect(result).toEqual({
      _version: 2,
      canvasSessionId: 'canvas-generated',
      canvasDiscardedQueueItems: [],
      areThumbnailsVisible: true,
    });
  });

  it('migrates v1 persisted state while preserving existing session data', () => {
    expect(migrate).toBeDefined();

    const result = migrate?.({
      _version: 1,
      canvasSessionId: 'canvas-v1',
      canvasDiscardedQueueItems: [3, 5],
    });

    expect(result).toEqual({
      _version: 2,
      canvasSessionId: 'canvas-v1',
      canvasDiscardedQueueItems: [3, 5],
      areThumbnailsVisible: true,
    });
  });
});
