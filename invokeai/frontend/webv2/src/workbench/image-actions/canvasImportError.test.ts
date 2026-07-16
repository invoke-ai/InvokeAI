import { describe, expect, it, vi } from 'vitest';

import { recordCanvasImportError } from './canvasImportError';

describe('recordCanvasImportError', () => {
  it('dispatches one localized diagnostic error and no notice', () => {
    const dispatch = vi.fn();

    recordCanvasImportError({
      dispatch,
      error: new Error('resize backend failed'),
      localizedMessage: "Couldn't add images to canvas",
      projectId: 'project-1',
    });

    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith({
      area: 'image-actions',
      context: { error: 'resize backend failed' },
      message: "Couldn't add images to canvas",
      namespace: 'gallery',
      projectId: 'project-1',
      type: 'recordError',
    });
    expect(dispatch.mock.calls.some(([action]) => action.type === 'recordNotice')).toBe(false);
  });
});
