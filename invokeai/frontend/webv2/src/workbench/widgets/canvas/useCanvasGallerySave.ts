import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';

import { saveCanvasToGallery, type CanvasGallerySaveRegion } from '@workbench/canvas-operations/saveCanvasToGallery';
import { useNotify } from '@workbench/useNotify';
import { useWorkbenchStore } from '@workbench/WorkbenchContext';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getCanvasGallerySaveErrorAction, withMatchingCanvasProject } from './canvasGallerySaveState';

type CanvasGallerySaveEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'lifecycle' | 'projectId'>;

export const useCanvasGallerySave = (
  engine: CanvasGallerySaveEngine | null
): { isSaving: boolean; save: (region: CanvasGallerySaveRegion) => Promise<void> } => {
  const { t } = useTranslation();
  const notify = useNotify();
  const store = useWorkbenchStore();
  const isSavingRef = useRef(false);
  const [isSaving, setIsSaving] = useState(false);

  const save = useCallback(
    async (region: CanvasGallerySaveRegion): Promise<void> => {
      if (isSavingRef.current) {
        return;
      }

      const project = store.getSnapshot().activeProject;

      await withMatchingCanvasProject(engine, project.id, async (matchedEngine) => {
        isSavingRef.current = true;
        setIsSaving(true);

        try {
          const result = await saveCanvasToGallery({ engine: matchedEngine, project, region });

          if (result.status === 'saved') {
            store.dispatch({ projectId: project.id, type: 'touchGalleryImagesRefresh' });
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
          store.dispatch(getCanvasGallerySaveErrorAction(error, project.id, t('widgets.canvas.contextMenu.saveError')));
        } finally {
          isSavingRef.current = false;
          setIsSaving(false);
        }
      });
    },
    [engine, notify, store, t]
  );

  return { isSaving, save };
};
