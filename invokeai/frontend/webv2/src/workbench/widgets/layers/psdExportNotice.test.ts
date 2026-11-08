import { describe, expect, it } from 'vitest';

import { getPsdExportNotice } from './psdExportNotice';

describe('getPsdExportNotice', () => {
  it.each([
    ['not-ready', 'widgets.layers.groupActions.exportNotReady'],
    ['too-large', 'widgets.layers.groupActions.exportTooLarge'],
    ['nothing', 'widgets.layers.groupActions.exportNothing'],
    ['over-budget', 'widgets.layers.groupActions.exportOverBudget'],
    ['stale', 'widgets.layers.groupActions.exportStale'],
    ['aborted', 'widgets.layers.groupActions.exportAborted'],
  ] as const)('maps %s to a user-visible typed notice', (result, key) => {
    expect(getPsdExportNotice(result)).toEqual({ key });
  });

  it('names the layer that could not be read', () => {
    expect(getPsdExportNotice({ layerId: 'layer-7', layerName: 'Sky', status: 'unreadable' })).toEqual({
      key: 'widgets.layers.groupActions.exportUnreadable',
      options: { name: 'Sky' },
    });
  });

  it('has nothing to say once exported', () => {
    expect(getPsdExportNotice('exported')).toBeNull();
  });
});
