import type { PsdExportResult } from '@workbench/canvas-engine/api';

export type PsdExportNoticeKey =
  | 'widgets.layers.groupActions.exportNotReady'
  | 'widgets.layers.groupActions.exportTooLarge'
  | 'widgets.layers.groupActions.exportNothing'
  | 'widgets.layers.groupActions.exportOverBudget'
  | 'widgets.layers.groupActions.exportStale'
  | 'widgets.layers.groupActions.exportAborted';

export const getPsdExportNoticeKey = (result: PsdExportResult): PsdExportNoticeKey | null => {
  switch (result) {
    case 'exported':
      return null;
    case 'not-ready':
      return 'widgets.layers.groupActions.exportNotReady';
    case 'too-large':
      return 'widgets.layers.groupActions.exportTooLarge';
    case 'nothing':
      return 'widgets.layers.groupActions.exportNothing';
    case 'over-budget':
      return 'widgets.layers.groupActions.exportOverBudget';
    case 'stale':
      return 'widgets.layers.groupActions.exportStale';
    case 'aborted':
      return 'widgets.layers.groupActions.exportAborted';
  }
};
