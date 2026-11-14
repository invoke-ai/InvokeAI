import type { AccountScope } from '@platform/state/accountLifecycle';

import { createItemId } from '@platform/browser/randomUuid';
import { createLogger } from '@platform/logging/logger';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useState } from 'react';
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

/**
 * Runs the upload of files dropped on a surface (Canvas, Preview), loading its code on demand. While one is under way
 * for the open project, a further drop on the surface is refused with `pendingKey`'s notice, so a slow upload is never
 * doubled; past a moment a progress notice (`uploadingKey`, with the file count) shows until it ends.
 * The drop's account, project and gallery values are captured at the drop, so nothing that changes while the code
 * loads or the files upload can retarget it. Code that cannot load (offline, or replaced by a newer deploy) is
 * reported as an upload that did not start; `runFailedMessage` names a failure of the work itself.
 */
export const useDroppedFileUpload = ({ pendingKey, uploadingKey }: { pendingKey: string; uploadingKey: string }) => {
  const { t } = useTranslation();
  const { notifications } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  // The project an upload is under way for; another project's drops go ahead.
  const [pendingProjectId, setPendingProjectId] = useState<string | null>(null);

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
      if (pendingProjectId === projectId) {
        notifications.add({ kind: 'info', title: t(pendingKey) });
        return;
      }
      const context: DroppedFileUploadContext = {
        galleryValues: getResolvedProjectWidgetValues(activeProject, 'gallery', scopedDefaults),
        owner: captureAccountScope(),
        projectId,
      };
      // The notice says what happened in the user's words; the cause goes to the log, not into the notice.
      const report = (message: string, error: unknown) => {
        if (isAccountScopeCurrent(context.owner)) {
          uploadLogger.error({ context: { files: files.length, projectId }, error, message });
          notifications.add({ kind: 'error', title: message });
        }
      };
      setPendingProjectId(projectId);
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
      void load()
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
          setPendingProjectId((current) => (current === projectId ? null : current));
        });
    },
    [notifications, pendingKey, pendingProjectId, queries, t, uploadingKey]
  );
};
