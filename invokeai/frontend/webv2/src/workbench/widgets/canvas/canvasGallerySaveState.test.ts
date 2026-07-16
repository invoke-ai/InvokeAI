import { describe, expect, it, vi } from 'vitest';

import { getCanvasGallerySaveErrorAction, withMatchingCanvasProject } from './canvasGallerySaveState';

describe('withMatchingCanvasProject', () => {
  it('does not run save, refresh, or error effects for a mismatched engine and project', () => {
    const save = vi.fn();
    const refresh = vi.fn();
    const recordError = vi.fn();

    const result = withMatchingCanvasProject({ projectId: 'engine-project' }, 'active-project', () => {
      save();
      refresh();
      recordError();
    });

    expect(result).toBeUndefined();
    expect(save).not.toHaveBeenCalled();
    expect(refresh).not.toHaveBeenCalled();
    expect(recordError).not.toHaveBeenCalled();
  });
});

describe('getCanvasGallerySaveErrorAction', () => {
  it('maps a save failure to one localized record with diagnostic context', () => {
    expect(
      getCanvasGallerySaveErrorAction(new Error('backend exploded'), 'project-1', "Couldn't save to gallery")
    ).toEqual({
      area: 'canvas-save-to-gallery',
      context: { error: 'backend exploded' },
      message: "Couldn't save to gallery",
      namespace: 'canvas',
      projectId: 'project-1',
      type: 'recordError',
    });
  });
});
