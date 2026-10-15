import { describe, expect, it } from 'vitest';

import { getPsdExportNoticeKey } from './psdExportNotice';

describe('getPsdExportNoticeKey', () => {
  it.each([
    ['not-ready', 'widgets.layers.groupActions.exportNotReady'],
    ['too-large', 'widgets.layers.groupActions.exportTooLarge'],
    ['nothing', 'widgets.layers.groupActions.exportNothing'],
    ['over-budget', 'widgets.layers.groupActions.exportOverBudget'],
    ['stale', 'widgets.layers.groupActions.exportStale'],
    ['aborted', 'widgets.layers.groupActions.exportAborted'],
    ['exported', null],
  ] as const)('maps %s to a user-visible typed notice', (result, expected) => {
    expect(getPsdExportNoticeKey(result)).toBe(expected);
  });
});
