import { describe, expect, it } from 'vitest';

import { getCanvasImportNotice } from './canvasImportNotice';

describe('getCanvasImportNotice', () => {
  it('maps a complete import to a localized success notice', () => {
    expect(
      getCanvasImportNotice({ failedImageNames: [], layerIds: ['layer-1', 'layer-2'], status: 'imported' })
    ).toEqual({
      kind: 'success',
      options: { count: 2 },
      titleKey: 'widgets.canvas.import.success',
    });
  });

  it('maps one successful partial import with the exact pluralization interpolation', () => {
    expect(
      getCanvasImportNotice({
        failedImageNames: ['failed-1.png', 'failed-2.png'],
        layerIds: ['layer-1'],
        status: 'imported',
      })
    ).toEqual({
      kind: 'info',
      options: { count: 1, failedCount: 2, successCount: 1 },
      titleKey: 'widgets.canvas.import.partial',
    });
  });

  it('maps multiple successful partial imports with the exact pluralization interpolation', () => {
    expect(
      getCanvasImportNotice({
        failedImageNames: ['failed.png'],
        layerIds: ['layer-1', 'layer-2'],
        status: 'imported',
      })
    ).toEqual({
      kind: 'info',
      options: { count: 2, failedCount: 1, successCount: 2 },
      titleKey: 'widgets.canvas.import.partial',
    });
  });

  it.each([
    ['blocked', 'info', 'widgets.canvas.import.blocked'],
    ['empty', 'info', 'widgets.canvas.import.empty'],
    ['stale-document', 'error', 'widgets.canvas.import.staleDocument'],
    ['stale-project', 'error', 'widgets.canvas.import.staleProject'],
  ] as const)('maps %s to the expected localized notice', (status, kind, titleKey) => {
    expect(getCanvasImportNotice({ status })).toEqual({ kind, titleKey });
  });
});
