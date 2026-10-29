import type { AdjustmentRefusal, PsdExportResult } from '@workbench/canvas-engine/api';

export type PsdExportNoticeKey =
  | 'widgets.layers.groupActions.exportNotReady'
  | 'widgets.layers.groupActions.exportTooLarge'
  | 'widgets.layers.groupActions.exportNothing'
  | 'widgets.layers.groupActions.exportOverBudget'
  | 'widgets.layers.groupActions.exportStale'
  | 'widgets.layers.groupActions.exportUnreadable'
  | 'widgets.layers.groupActions.exportAborted';

/** A notice's translation key and interpolation; an unreadable layer is named. */
export interface PsdExportNotice {
  readonly key: PsdExportNoticeKey;
  readonly options?: { readonly name: string };
}

/** The notice for a non-refusal result; an `AdjustmentRefusal` is worded by `describeAdjustmentRefusal`. */
export const getPsdExportNotice = (result: Exclude<PsdExportResult, AdjustmentRefusal>): PsdExportNotice | null => {
  if (typeof result === 'object') {
    return { key: 'widgets.layers.groupActions.exportUnreadable', options: { name: result.layerName } };
  }
  switch (result) {
    case 'exported':
      return null;
    case 'not-ready':
      return { key: 'widgets.layers.groupActions.exportNotReady' };
    case 'too-large':
      return { key: 'widgets.layers.groupActions.exportTooLarge' };
    case 'nothing':
      return { key: 'widgets.layers.groupActions.exportNothing' };
    case 'over-budget':
      return { key: 'widgets.layers.groupActions.exportOverBudget' };
    case 'stale':
      return { key: 'widgets.layers.groupActions.exportStale' };
    case 'aborted':
      return { key: 'widgets.layers.groupActions.exportAborted' };
  }
};
