import type { GenerationIssue } from '@features/generation/contracts';
import type { InvocationSourceId, InvocationValidationReason, ResultDestination } from '@workbench/invocationContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { getWidgetTypeIdForSourceId } from '@workbench/graphWidgets';

/** Where an issue is fixed, so Invoke's issue list can lead there. */
export type InvocationIssueTarget =
  /** The server is unreachable: nothing to click, it resolves on reconnect. */
  | { kind: 'connection' }
  /** A Generate setting: open the Generate panel at its section, or apply its one-step fix. */
  | { kind: 'generation'; issue: GenerationIssue }
  /** The positive prompt could not be expanded. */
  | { kind: 'prompt' }
  | { kind: 'layer'; layerId: string }
  | { kind: 'workflowNode'; nodeId: string }
  /** The source's widget is not in the layout: add it, or run another source. */
  | { kind: 'mountSource'; sourceId: InvocationSourceId }
  /** The destination cannot take this source's results. */
  | { kind: 'destination'; destination: ResultDestination }
  /** Fixed somewhere in a widget: open it. */
  | { kind: 'widget'; widgetId: WidgetTypeId };

/** Why Invoke cannot run, worded for the user, and where it is fixed. */
export interface InvocationIssue {
  message: string;
  target: InvocationIssueTarget;
}

/** Where a route reason is fixed; reasons with no finer target lead to the source's own widget. */
export const getInvocationIssueTarget = (
  reason: InvocationValidationReason,
  sourceId: InvocationSourceId
): InvocationIssueTarget => {
  if (typeof reason === 'object') {
    if ('controlLayerIssue' in reason) {
      return { kind: 'layer', layerId: reason.controlLayerIssue.layerId };
    }
    if ('generationIssue' in reason) {
      return { issue: reason.generationIssue, kind: 'generation' };
    }
    if ('routeIssue' in reason) {
      return reason.routeIssue.kind === 'sourceNotMounted'
        ? { kind: 'mountSource', sourceId: reason.routeIssue.sourceId }
        : reason.routeIssue.kind === 'videoNeedsGallery'
          ? { destination: 'gallery', kind: 'destination' }
          : { kind: 'widget', widgetId: 'canvas' };
    }
    if ('workflowIssue' in reason && reason.workflowIssue.nodeId !== undefined) {
      return { kind: 'workflowNode', nodeId: reason.workflowIssue.nodeId };
    }
  }

  return { kind: 'widget', widgetId: getWidgetTypeIdForSourceId(sourceId) };
};
