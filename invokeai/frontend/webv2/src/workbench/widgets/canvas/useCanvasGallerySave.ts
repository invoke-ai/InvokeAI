import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';

import { invalidateGallery } from '@features/gallery/queries';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { saveCanvasToGallery, type CanvasGallerySaveRegion } from '@workbench/canvas-operations/api';
import { useNotify } from '@workbench/useNotify';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getCanvasGallerySaveErrorAction, withMatchingCanvasProject } from './canvasGallerySaveState';

type CanvasGallerySaveEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'lifecycle' | 'projectId'>;

export const useCanvasGallerySave = (
  engine: CanvasGallerySaveEngine | null
): { isSaving: boolean; save: (region: CanvasGallerySaveRegion) => Promise<void> } => {
  const { t } = useTranslation();
  const notify = useNotify();
  const queries = useWorkbenchQueries();
  const { notifications } = useWorkbenchCommands();
  const queryClient = useQueryClient();
  const isSavingRef = useRef(false);
  const [isSaving, setIsSaving] = useState(false);

  const save = useCallback(
    async (region: CanvasGallerySaveRegion): Promise<void> => {
      if (isSavingRef.current) {
        return;
      }

      const project = queries.getSnapshot().activeProject;
      const owner = captureAccountScope();

      await withMatchingCanvasProject(engine, project.id, async (matchedEngine) => {
        isSavingRef.current = true;
        setIsSaving(true);

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

          notifications.reportError(
            getCanvasGallerySaveErrorAction(error, project.id, t('widgets.canvas.contextMenu.saveError'))
          );
        } finally {
          isSavingRef.current = false;
          setIsSaving(false);
        }
      });
    },
    [engine, notifications, notify, queries, queryClient, t]
  );

  return { isSaving, save };
};
