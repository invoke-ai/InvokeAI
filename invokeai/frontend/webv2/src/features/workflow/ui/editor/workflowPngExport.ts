import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';
import type { TFunction } from 'i18next';

import { createExternalStore } from '@platform/state/externalStore';
import { flushSync } from 'react-dom';

import { getWorkflowFlowInstance } from './flowInstanceStore';

/**
 * "Download workflow as PNG", shared by the editor toolbar and the widget's actions menu. The mounted editor lends the
 * offscreen view the image is drawn from; while none is mounted (the form alone is shown) the download is unavailable.
 * One export runs at a time from either entry point.
 */
export interface WorkflowPngExportHost {
  /** Mounts the offscreen full-graph view and returns its flow element. */
  prepare: () => Promise<HTMLElement>;
  /** Unmounts that view. */
  complete: () => void;
}

const exportStore = createExternalStore<{ host: WorkflowPngExportHost | null; isExporting: boolean }>({
  host: null,
  isExporting: false,
});

/** Registered by the editor for as long as it is mounted; the returned function releases it. */
export const registerWorkflowPngExportHost = (host: WorkflowPngExportHost): (() => void) => {
  exportStore.patchSnapshot({ host });

  return () => {
    if (exportStore.getSnapshot().host === host) {
      exportStore.patchSnapshot({ host: null });
    }
  };
};

export const useWorkflowPngExportState = () =>
  exportStore.useSelector((snapshot) => ({ isAvailable: snapshot.host !== null, isExporting: snapshot.isExporting }));

/**
 * Draws the active workflow to a PNG and downloads it, reporting refusals and reductions. The drawing loads on first
 * use, so this store stays small enough for the menus that start an export to load it at startup. Resolves once the
 * busy state has been cleared and rendered, so a caller can hand focus back to a control the busy state disabled.
 */
export const exportWorkflowPng = async ({
  notifications,
  t,
  workflowName,
}: {
  notifications: WorkflowUiAdapter['notifications'];
  t: TFunction;
  workflowName: string;
}): Promise<void> => {
  const { host, isExporting } = exportStore.getSnapshot();
  const flow = getWorkflowFlowInstance();

  if (isExporting || !host || !flow) {
    return;
  }

  exportStore.patchSnapshot({ isExporting: true });
  try {
    const { runWorkflowPngExport } = await import('./workflowPngExportRun');
    await runWorkflowPngExport({ bounds: flow.getNodesBounds(flow.getNodes()), host, notifications, t, workflowName });
  } catch {
    notifications.error(t('widgets.workflow.exportImageFailed'));
  } finally {
    flushSync(() => exportStore.patchSnapshot({ isExporting: false }));
    host.complete();
  }
};
