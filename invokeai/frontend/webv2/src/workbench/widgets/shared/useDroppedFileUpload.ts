import type { AccountScope } from '@platform/state/accountLifecycle';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { createItemId } from '@platform/browser/randomUuid';
import { createLogger } from '@platform/logging/logger';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/** What a dropped upload is fenced to, captured at the drop. */
export interface DroppedFileUploadContext {
  /** The project's resolved gallery values at the drop, which name its auto-add board. */
  galleryValues: Record<string, unknown>;
  owner: AccountScope;
  projectId: string;
}

/** A quick upload finishes unnoticed; a slower one shows a progress notice until it does, as deletion checks do. */
const UPLOADING_NOTICE_DELAY_MS = 400;

const uploadLogger = createLogger({ area: 'file-drop-upload', namespace: 'gallery' });

// Uploads under way, by account lifetime, surface and project. Module-held: a surface's host is rebuilt when the
// project changes (a floating window is keyed by project), and the new instance must still see the upload. The account
// epoch keeps a stale lifetime's upload from holding the new one, and each key admits one upload, so the upload that
// added it is the only one that removes it.
const pendingUploads = new Set<string>();

/**
 * Runs the upload of files dropped on a surface (Canvas, Preview), loading its code on demand. Each surface, named by
 * its widget (`surface`; neither allows a second instance per project), runs one upload per project at a time: while
 * one is under way for the open project, a further drop on that surface is refused with `pendingKey`'s notice, even
 * after the surface remounts, so a slow upload is never doubled; other projects and surfaces go ahead. Past a moment a
 * progress notice (`uploadingKey`, with the file count) shows until the upload ends. The drop's account, project and
 * gallery values are captured at the drop, so nothing that changes while the code loads or the files upload can
 * retarget it. Code that cannot load (offline, or replaced by a newer deploy) is reported as an upload that did not
 * start; `runFailedMessage` names a failure of the work itself.
 */
export const useDroppedFileUpload = ({
  pendingKey,
  surface,
  uploadingKey,
}: {
  pendingKey: string;
  surface: WidgetTypeId;
  uploadingKey: string;
}) => {
  const { t } = useTranslation();
  const { notifications } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();

  return useCallback(
    <Module>(
      files: File[],
      {
        load,
        run,
        runFailedMessage,
      }: {
        load: () => Promise<Module>;
        run: (module: Module, context: DroppedFileUploadContext) => Promise<void>;
        runFailedMessage: string;
      }
    ) => {
      const { activeProject, scopedDefaults } = queries.getSnapshot();
      const projectId = activeProject.id;
      const owner = captureAccountScope();
      const uploadKey = JSON.stringify([owner.epoch, surface, projectId]);
      if (pendingUploads.has(uploadKey)) {
        notifications.add({ kind: 'info', title: t(pendingKey) });
        return;
      }
      const context: DroppedFileUploadContext = {
        galleryValues: getResolvedProjectWidgetValues(activeProject, 'gallery', scopedDefaults),
        owner,
        projectId,
      };
      // The notice says what happened in the user's words; the cause goes to the log, not into the notice.
      const report = (message: string, error: unknown) => {
        if (isAccountScopeCurrent(context.owner)) {
          uploadLogger.error({ context: { files: files.length, projectId }, error, message });
          notifications.add({ kind: 'error', title: message });
        }
      };
      pendingUploads.add(uploadKey);
      const toastId = createItemId('file-drop-upload');
      const timer = setTimeout(() => {
        // Infinite: the notice goes when the upload ends, however it ends.
        toaster.create({
          duration: Number.POSITIVE_INFINITY,
          id: toastId,
          title: t(uploadingKey, { count: files.length }),
          type: 'loading',
        });
      }, UPLOADING_NOTICE_DELAY_MS);
      // From a settled promise, so a load that throws at once still ends the upload below.
      void Promise.resolve()
        .then(load)
        .then(
          (module) =>
            isAccountScopeCurrent(context.owner)
              ? run(module, context).catch((error: unknown) => report(runFailedMessage, error))
              : undefined,
          (error: unknown) => report(t('widgets.gallery.uploadFailed', { failed: files.length }), error)
        )
        .finally(() => {
          clearTimeout(timer);
          toaster.dismiss(toastId);
          pendingUploads.delete(uploadKey);
        });
    },
    [notifications, pendingKey, queries, surface, t, uploadingKey]
  );
};
