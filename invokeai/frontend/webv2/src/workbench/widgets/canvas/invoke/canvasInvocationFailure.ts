import type { ComposeForGenerationResult } from '@workbench/canvas-operations/api';

import { describeCanvasInputIssue, type CanvasInputIssue } from '@workbench/canvasGenerationInputs';

export type CanvasInvocationFailure =
  | { kind: 'canvas_input'; issue: CanvasInputIssue }
  | { kind: 'capture'; result: Exclude<ComposeForGenerationResult, { status: 'ok' }> }
  | { kind: 'no_model' }
  | { kind: 'canvas_not_open' }
  | { kind: 'error'; message: string };

export interface CanvasInvocationNotice {
  title: string;
  message: string;
}

type Translate = (key: string, options?: Record<string, unknown>) => string;

const describeAdjustmentRefusal = (t: Translate, reason: 'error' | 'over-budget' | 'too-large', name: string) => {
  switch (reason) {
    case 'error':
      return t('widgets.canvas.invoke.adjustmentRefused.error', { name });
    case 'over-budget':
      return t('widgets.canvas.invoke.adjustmentRefused.overBudget', { name });
    case 'too-large':
      return t('widgets.canvas.invoke.adjustmentRefused.tooLarge', { name });
  }
};

const describeCapture = (t: Translate, result: Extract<CanvasInvocationFailure, { kind: 'capture' }>['result']) => {
  switch (result.status) {
    case 'no-document':
      return t('widgets.canvas.invoke.noDocument');
    case 'stale':
      return t('widgets.canvas.invoke.stale');
    case 'adjustment-refused':
      return describeAdjustmentRefusal(t, result.reason, result.ownerName);
    case 'unreadable':
      return t('widgets.canvas.invoke.unreadable', { name: result.layerName });
    case 'not-ready':
      return t('widgets.canvas.invoke.notReady');
    case 'over-budget':
      return t('widgets.canvas.invoke.overBudget');
    case 'aborted':
      return t('widgets.canvas.invoke.captureFailed', { status: result.status });
  }
};

export const describeCanvasInvocationFailure = (
  t: Translate,
  failure: CanvasInvocationFailure
): CanvasInvocationNotice => ({
  message:
    failure.kind === 'canvas_input'
      ? describeCanvasInputIssue(t, failure.issue)
      : failure.kind === 'capture'
        ? describeCapture(t, failure.result)
        : failure.kind === 'no_model'
          ? t('widgets.canvas.invoke.noModel')
          : failure.kind === 'canvas_not_open'
            ? t('widgets.canvas.invoke.notOpen')
            : failure.message,
  title: t('widgets.canvas.invoke.failed'),
});
