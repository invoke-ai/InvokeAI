import type { ToastAction } from '@platform/ui/toaster';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { invalidateGallery, invalidateGalleryBoardDestinations } from '@features/gallery/queries';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createKeyedTransientStore } from '@platform/state/externalStore';
import { ApiError } from '@platform/transport/http';
import { useQueryClient } from '@tanstack/react-query';
import { saveCanvasToGallery, type CanvasGallerySaveRegion } from '@workbench/canvas-operations/api';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { useNotify } from '@workbench/useNotify';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { getCanvasGallerySaveErrorAction, withMatchingCanvasProject } from './canvasGallerySaveState';
import { ACTION_TOAST_DURATION_MS, announce, describeBoard } from './gallerySaveToast';
import { describeAdjustmentRefusal } from './useStructuralCommit';

type CanvasGallerySaveEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'lifecycle' | 'projectId'>;

/** Projects with a save in flight: the header button and the context menu share the gate and the busy state. */
const savingProjects = createKeyedTransientStore<string, true>();
registerAccountOwnedResource({ clear: () => savingProjects.clear(), name: 'canvas-gallery-saves' });

export const useCanvasGallerySave = (
  engine: CanvasGallerySaveEngine | null
): { isSaving: boolean; save: (region: CanvasGallerySaveRegion) => Promise<void> } => {
  const { t } = useTranslation();
  const notify = useNotify();
  const queries = useWorkbenchQueries();
  const { notifications } = useWorkbenchCommands();
  const queryClient = useQueryClient();
  const findGalleryItem = useFindGalleryItem();
  const isSaving = savingProjects.useValue(engine?.projectId ?? '') === true;

  const save = useCallback(
    async (region: CanvasGallerySaveRegion): Promise<void> => {
      const project = queries.getSnapshot().activeProject;
      if (savingProjects.get(project.id)) {
        return;
      }
      const owner = captureAccountScope();

      await withMatchingCanvasProject(engine, project.id, async (matchedEngine) => {
        savingProjects.set(project.id, true);

        try {
          const result = await saveCanvasToGallery({ engine: matchedEngine, project, region });

          assertAccountScopeCurrent(owner);
          if (result.status === 'saved') {
            void invalidateGallery(queryClient, owner);
            const board = await describeBoard(queryClient, project, result.boardId, t);

            if (!isAccountScopeCurrent(owner)) {
              return;
            }

            // Reveals and selects the image in a Gallery panel (its board, Media, cleared filters) and leaves the
            // center on the Canvas, while the project that saved it is still open.
            const showInGallery: ToastAction = {
              label: t('widgets.canvas.staging.showInGallery'),
              onClick: () => {
                if (isAccountScopeCurrent(owner) && queries.isActiveProject(project.id)) {
                  findGalleryItem({ kind: 'image', name: result.imageName }, { revealPreview: false });
                }
              },
            };
            const showActions = queries.isActiveProject(project.id) ? [showInGallery] : [];

            announce(owner, {
              actions: showActions,
              description: t('widgets.canvas.staging.savedDescription', { name: result.imageName }),
              ...(showActions.length > 0 ? { duration: ACTION_TOAST_DURATION_MS } : {}),
              title:
                board.kind === 'named'
                  ? t('widgets.canvas.staging.savedToBoard', { board: board.label })
                  : t('widgets.canvas.contextMenu.saved'),
              type: 'success',
            });
          } else if (result.status === 'empty') {
            notify.info(t('widgets.canvas.contextMenu.empty'));
          } else if (result.status === 'stale') {
            notify.info(t('widgets.canvas.contextMenu.stale'));
          } else if (result.status === 'adjustment-refused') {
            notify.error(t('widgets.canvas.contextMenu.saveError'), describeAdjustmentRefusal(result, t));
          } else if (result.status === 'unreadable') {
            notify.error(t('widgets.canvas.contextMenu.saveError'), t('widgets.canvas.contextMenu.unreadable'));
          } else {
            notify.info(t('widgets.canvas.contextMenu.notReady'));
          }
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          // The board refused the save (403) or is gone (404): the listing the board was chosen from is read again
          // at once, so a save that follows the read is routed by it.
          if (error instanceof ApiError && (error.status === 403 || error.status === 404)) {
            void invalidateGalleryBoardDestinations(queryClient, owner);
          }
          notifications.reportError(
            getCanvasGallerySaveErrorAction(error, project.id, t('widgets.canvas.contextMenu.saveError'))
          );
        } finally {
          savingProjects.delete(project.id);
        }
      });
    },
    [engine, findGalleryItem, notifications, notify, queries, queryClient, t]
  );

  return { isSaving, save };
};
