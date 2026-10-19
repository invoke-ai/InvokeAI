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
import { useNotify } from '@workbench/useNotify';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { getCanvasGallerySaveErrorAction, withMatchingCanvasProject } from './canvasGallerySaveState';

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
            notify.success(
              t('widgets.canvas.contextMenu.saved'),
              t('widgets.canvas.contextMenu.savedDescription', { name: result.imageName })
            );
          } else if (result.status === 'empty') {
            notify.info(t('widgets.canvas.contextMenu.empty'));
          } else if (result.status === 'stale') {
            notify.info(t('widgets.canvas.contextMenu.stale'));
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
    [engine, notifications, notify, queries, queryClient, t]
  );

  return { isSaving, save };
};
