import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';
import type { Rect } from '@xyflow/react';
import type { TFunction } from 'i18next';

import type { WorkflowPngExportHost } from './workflowPngExport';

import { exportWorkflowAsPng, preflightWorkflowImageExport } from './workflowImageExport';

const waitForExportFrame = () =>
  new Promise<void>((resolve) => {
    if (document.visibilityState === 'hidden') {
      window.setTimeout(resolve, 0);
    } else {
      window.requestAnimationFrame(() => resolve());
    }
  });

/**
 * The download `exportWorkflowPng` starts: refuses from the live editor's bounds before mounting the whole graph a
 * second time, draws it from the host's offscreen view, and reports the outcome. Unexpected failures reject.
 */
export const runWorkflowPngExport = async ({
  bounds,
  host,
  notifications,
  t,
  workflowName,
}: {
  bounds: Rect;
  host: WorkflowPngExportHost;
  notifications: WorkflowUiAdapter['notifications'];
  t: TFunction;
  workflowName: string;
}): Promise<void> => {
  let outcome = preflightWorkflowImageExport(bounds);
  if (!outcome) {
    const flowElement = await host.prepare();
    await waitForExportFrame();
    await waitForExportFrame();
    outcome = await exportWorkflowAsPng({
      bounds,
      fallbackWorkflowName: t('widgets.workflow.untitled'),
      flowElement,
      workflowName,
    });
  }
  if (outcome.status === 'busy') {
    notifications.info(t('widgets.workflow.exportImageBusy'));
  } else if (outcome.status === 'too-large') {
    notifications.error(t('widgets.workflow.exportImageTooLarge'), t('widgets.workflow.exportImageTooLargeDetail'));
  } else if (outcome.status === 'exported' && outcome.reduced) {
    notifications.info(
      t('widgets.workflow.exportImageReduced'),
      t('widgets.workflow.exportImageReducedDetail', { height: outcome.height, width: outcome.width })
    );
  }
};
