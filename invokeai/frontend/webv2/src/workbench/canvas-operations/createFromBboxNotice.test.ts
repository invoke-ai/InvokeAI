import { describe, expect, it } from 'vitest';

import { getCreateFromBboxNotice } from './createFromBboxNotice';

describe('getCreateFromBboxNotice', () => {
  it.each([
    ['raster', 'widgets.canvas.import.raster'],
    ['control', 'widgets.canvas.import.control'],
    ['regional-reference', 'widgets.canvas.import.regionalReference'],
  ] as const)('labels a created %s destination with its import label key', (destination, labelKey) => {
    expect(getCreateFromBboxNotice({ destination, layerIds: ['layer-1'], status: 'created' })).toEqual({
      kind: 'success',
      options: { destination: labelKey },
      titleKey: 'widgets.canvas.contextMenu.createdFromBbox',
    });
  });

  it.each([
    ['empty', 'info', 'widgets.canvas.contextMenu.empty'],
    ['stale', 'info', 'widgets.canvas.contextMenu.stale'],
    ['not-ready', 'info', 'widgets.canvas.contextMenu.notReady'],
    ['over-budget', 'info', 'widgets.canvas.contextMenu.notReady'],
    ['blocked', 'info', 'widgets.canvas.import.blocked'],
    ['stale-document', 'error', 'widgets.canvas.import.staleDocument'],
    ['stale-project', 'error', 'widgets.canvas.import.staleProject'],
  ] as const)('maps %s to a %s notice', (status, kind, titleKey) => {
    expect(getCreateFromBboxNotice({ status })).toEqual({ kind, titleKey });
  });
});
