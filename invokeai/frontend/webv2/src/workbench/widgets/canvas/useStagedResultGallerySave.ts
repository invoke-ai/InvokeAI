import type { AccountScope } from '@platform/state/accountLifecycle';
import type { Project } from '@workbench/projectContracts';

import { invalidateGallery } from '@features/gallery/queries';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createKeyedTransientStore } from '@platform/state/externalStore';
import { ACTION_TOAST_DURATION_MS, createActionToast, type ToastAction } from '@platform/ui/toaster';
import { useQueryClient } from '@tanstack/react-query';
import {
  retryStagedResultBoard,
  saveStagedResultToGallery,
  type SaveStagedResultOutcome,
} from '@workbench/canvas-operations/api';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { describeBoard } from './gallerySaveToast';

/** Staged images with a save or board retry in flight; the bar and the thumbnail menu share the gate. */
const savingImages = createKeyedTransientStore<string, true>();
registerAccountOwnedResource({ clear: () => savingImages.clear(), name: 'staged-result-gallery-saves' });

/** What a save, or a board retry, belongs to: the account and project current when the user asked for it. */
interface SaveOrigin {
  owner: AccountScope;
  project: Project;
}

/**
 * Save a staged canvas result to the Gallery and say where it went. The account and project are captured when the
 * save starts and every toast action is fenced to them: nothing runs for an account that has since signed out, and
 * "Show in Gallery" acts only while the project is still open. It reveals the item in a Gallery panel and leaves
 * the center (the Canvas under review) where it is.
 */
export const useStagedResultGallerySave = (
  imageName: string | null
): { isSaving: boolean; save: (imageName: string) => Promise<void> } => {
  const { t } = useTranslation();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const findGalleryItem = useFindGalleryItem();
  const isSaving = savingImages.useValue(imageName ?? '') === true;

  const save = useMemo(() => {
    const run = async (
      name: string,
      { owner, project }: SaveOrigin,
      attempt: () => Promise<SaveStagedResultOutcome>
    ): Promise<void> => {
      if (!isAccountScopeCurrent(owner) || savingImages.get(name)) {
        return;
      }

      const showInGallery: ToastAction = {
        label: t('widgets.canvas.staging.showInGallery'),
        onClick: () => {
          if (isAccountScopeCurrent(owner) && queries.isActiveProject(project.id)) {
            findGalleryItem({ kind: 'image', name }, { revealPreview: false });
          }
        },
      };

      savingImages.set(name, true);

      try {
        const outcome = await attempt();

        if (outcome.status === 'stale' || !isAccountScopeCurrent(owner)) {
          return;
        }

        // The listing, its paging and the board counts are server-owned; refetch rather than guess where it lands.
        void invalidateGallery(queryClient, owner);
        const board = await describeBoard(
          queryClient,
          project,
          queries.getSnapshot().scopedDefaults,
          outcome.boardId,
          t
        );

        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        const showActions = queries.isActiveProject(project.id) ? [showInGallery] : [];
        const description = t('widgets.canvas.staging.savedDescription', { name });

        if (outcome.status === 'saved') {
          const title =
            board.kind === 'named'
              ? t('widgets.canvas.staging.savedToBoard', { board: board.label })
              : t('widgets.canvas.staging.saved');

          createActionToast({
            signal: owner.signal,
            actions: showActions,
            description,
            ...(showActions.length > 0 ? { duration: ACTION_TOAST_DURATION_MS } : {}),
            title,
            type: 'success',
          });
          return;
        }

        if (board.kind === 'missing') {
          // Retrying cannot bring a deleted board back.
          createActionToast({
            signal: owner.signal,
            actions: showActions,
            description: t('widgets.canvas.staging.boardMissingDescription', { name }),
            ...(showActions.length > 0 ? { duration: ACTION_TOAST_DURATION_MS } : {}),
            title: t('widgets.canvas.staging.boardMissing'),
            type: 'warning',
          });
          return;
        }

        const retry: ToastAction = {
          label: t('common.retry'),
          onClick: () =>
            void run(name, { owner, project }, () =>
              retryStagedResultBoard({ boardId: outcome.boardId, imageName: name })
            ),
        };

        createActionToast({
          signal: owner.signal,
          actions: [retry, ...showActions],
          description: t('widgets.canvas.staging.boardFailedDescription', { name }),
          duration: ACTION_TOAST_DURATION_MS,
          title:
            board.kind === 'named'
              ? t('widgets.canvas.staging.boardFailed', { board: board.label })
              : t('widgets.canvas.staging.boardFailedUnnamed'),
          type: 'warning',
        });
      } catch {
        if (isAccountScopeCurrent(owner)) {
          createActionToast({
            actions: [],
            signal: owner.signal,
            title: t('widgets.canvas.staging.saveError'),
            type: 'error',
          });
        }
      } finally {
        savingImages.delete(name);
      }
    };

    return (name: string) => {
      const snapshot = queries.getSnapshot();
      const origin = { owner: captureAccountScope(), project: snapshot.activeProject };

      return run(name, origin, () =>
        saveStagedResultToGallery({ imageName: name, project: origin.project, scopedDefaults: snapshot.scopedDefaults })
      );
    };
  }, [findGalleryItem, queries, queryClient, t]);

  return { isSaving, save };
};
