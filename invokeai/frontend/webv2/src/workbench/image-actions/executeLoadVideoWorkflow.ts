import type { GalleryVideoItem } from '@features/gallery';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { galleryVideos } from '@features/gallery';
import { requestWorkflowDocumentLoad } from '@features/workflow/react';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/**
 * The video counterpart of `executeLoadImageWorkflow`: open the workflow editor before requesting replacement with
 * the workflow embedded in the video; false from `openWorkflowEditor` means no editor can consume it.
 */
export const executeLoadVideoWorkflow = async ({
  isProjectActive,
  item,
  notifications,
  openWorkflowEditor,
  t,
}: {
  /** Re-checked after the fetch: a store-resident request must not land on another project's editor. */
  isProjectActive: () => boolean;
  item: GalleryVideoItem;
  notifications: Pick<WorkbenchCommands['notifications'], 'add'>;
  openWorkflowEditor: () => boolean;
  t: TFunction;
}): Promise<void> => {
  const owner = captureAccountScope();

  try {
    const { workflow } = await galleryVideos.workflow(item.name, owner.signal);

    assertAccountScopeCurrent(owner);

    if (!workflow) {
      notifications.add({ kind: 'info', title: t('widgets.gallery.itemActions.loadWorkflow.missingVideo') });
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
      t('widgets.gallery.itemActions.loadWorkflow.loadedLabelVideo', { name: name || item.name })
    );
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    notifications.add({
      kind: 'error',
      message: toErrorMessage(error),
      title: t('widgets.gallery.itemActions.loadWorkflow.failedVideo'),
    });
  }
};
