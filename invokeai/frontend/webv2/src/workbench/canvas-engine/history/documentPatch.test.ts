import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { describe, expect, it, vi } from 'vitest';

import { createDocumentPatchEntry } from './documentPatch';

const forward: CanvasProjectMutation = { direction: 1, type: 'cycleStagedImage' };
const inverse: CanvasProjectMutation = { direction: -1, type: 'cycleStagedImage' };

describe('createDocumentPatchEntry', () => {
  it('dispatches inverse on undo and forward on redo', () => {
    const dispatch = vi.fn();
    const entry = createDocumentPatchEntry({ dispatch, forward, inverse, label: 'Cycle' });

    entry.undo();
    expect(dispatch).toHaveBeenNthCalledWith(1, inverse);
    entry.redo();
    expect(dispatch).toHaveBeenNthCalledWith(2, forward);
  });

  it('defaults bytes to a small nominal cost, overridable', () => {
    const dispatch = vi.fn();
    const entry = createDocumentPatchEntry({ dispatch, forward, inverse, label: 'Cycle' });
    expect(entry.bytes).toBe(HISTORY_ENTRY_OVERHEAD_BYTES);

    const heavier = createDocumentPatchEntry({ bytes: 4096, dispatch, forward, inverse, label: 'Cycle' });
    expect(heavier.bytes).toBe(4096);
    expect(heavier.label).toBe('Cycle');
  });

  it('retains media names embedded in an undoable removed layer', () => {
    const removedLayer = {
      type: 'addCanvasLayer',
      layer: { source: { imageName: 'removed.png', video_name: 'source.mp4' } },
    } as unknown as CanvasProjectMutation;
    const entry = createDocumentPatchEntry({
      dispatch: vi.fn(),
      forward,
      inverse: removedLayer,
      label: 'Remove layer',
    });
    expect(entry.heldAssetRefs).toEqual({ images: ['removed.png'], videos: ['source.mp4'] });
  });
});
