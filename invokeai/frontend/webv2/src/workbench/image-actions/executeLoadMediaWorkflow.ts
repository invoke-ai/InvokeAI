import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { galleryImages, galleryVideos } from '@features/gallery';
import { requestWorkflowDocumentLoad } from '@features/workflow/react';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

export interface WorkflowMediaRef {
  kind: 'image' | 'video';
  /** The image or video name the backend stores the embedded workflow under. */
  name: string;
}

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const fetchEmbeddedWorkflow = (media: WorkflowMediaRef, signal: AbortSignal) =>
  media.kind === 'image' ? galleryImages.workflow(media.name, signal) : galleryVideos.workflow(media.name, signal);

/**
 * Open the workflow editor before requesting replacement with the workflow embedded in an image or video; false
 * from `openWorkflowEditor` means no editor can consume it.
 */
export const executeLoadMediaWorkflow = async ({
  isProjectActive,
  media,
  notifications,
  openWorkflowEditor,
  t,
}: {
  /** Re-checked after the fetch: a store-resident request must not land on another project's editor. */
  isProjectActive: () => boolean;
  media: WorkflowMediaRef;
  notifications: Pick<WorkbenchCommands['notifications'], 'add'>;
  openWorkflowEditor: () => boolean;
  t: TFunction;
}): Promise<void> => {
  const owner = captureAccountScope();
  const suffix = media.kind === 'video' ? 'Video' : '';

  try {
    const { workflow } = await fetchEmbeddedWorkflow(media, owner.signal);

    assertAccountScopeCurrent(owner);

    if (!workflow) {
      notifications.add({ kind: 'info', title: t(`widgets.gallery.itemActions.loadWorkflow.missing${suffix}`) });
      return;
    }

    if (!isProjectActive()) {
      return;
    }

    if (!openWorkflowEditor()) {
      throw new Error(t('widgets.gallery.itemActions.loadWorkflow.editorUnavailable'));
    }

    const raw: unknown = JSON.parse(workflow);
    const name =
      typeof raw === 'object' && raw !== null && typeof (raw as { name?: unknown }).name === 'string'
        ? (raw as { name: string }).name
        : '';

    requestWorkflowDocumentLoad(
      raw,
      t(`widgets.gallery.itemActions.loadWorkflow.loadedLabel${suffix}`, { name: name || media.name })
    );
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    notifications.add({
      kind: 'error',
      message: toErrorMessage(error),
      title: t(`widgets.gallery.itemActions.loadWorkflow.failed${suffix}`),
    });
  }
};
