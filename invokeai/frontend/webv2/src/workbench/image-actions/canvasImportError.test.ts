import type { WorkbenchNotificationCommands } from '@workbench/workbenchStore';

import { describe, expect, it, vi } from 'vitest';

import { recordCanvasImportError } from './canvasImportError';

describe('recordCanvasImportError', () => {
  it('dispatches one localized diagnostic error and no notice', () => {
    const reportError = vi.fn();
    const notifications = { reportError } as unknown as WorkbenchNotificationCommands;

    recordCanvasImportError({
      notifications,
      error: new Error('resize backend failed'),
      localizedMessage: "Couldn't add images to canvas",
      projectId: 'project-1',
    });

    expect(reportError).toHaveBeenCalledTimes(1);
    expect(reportError).toHaveBeenCalledWith({
      area: 'image-actions',
      context: { error: 'resize backend failed' },
      message: "Couldn't add images to canvas",
      namespace: 'gallery',
      projectId: 'project-1',
    });
  });
});
