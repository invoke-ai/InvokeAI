import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';

import { saveCanvasToGallery, type CanvasGallerySaveRegion } from '@workbench/canvas-operations/saveCanvasToGallery';
import { useNotify } from '@workbench/useNotify';
import { useWorkbenchStore } from '@workbench/WorkbenchContext';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const useCanvasGallerySave = (
  engine: CanvasEngine | null
): { isSaving: boolean; save: (region: CanvasGallerySaveRegion) => Promise<void> } => {
  const { t } = useTranslation();
  const notify = useNotify();
  const store = useWorkbenchStore();
  const isSavingRef = useRef(false);
  const [isSaving, setIsSaving] = useState(false);

  const save = useCallback(
    async (region: CanvasGallerySaveRegion): Promise<void> => {
      if (!engine || isSavingRef.current) {
        return;
      }

      isSavingRef.current = true;
      setIsSaving(true);
      const project = store.getSnapshot().activeProject;

      try {
        const result = await saveCanvasToGallery({ engine, project, region });

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
        const message = toErrorMessage(error);
        store.dispatch({
          area: 'canvas-save-to-gallery',
          message,
          namespace: 'canvas',
          projectId: project.id,
          type: 'recordError',
        });
        notify.error(t('widgets.canvas.contextMenu.saveError'), message);
      } finally {
        isSavingRef.current = false;
        setIsSaving(false);
      }
    },
    [engine, notify, store, t]
  );

  return { isSaving, save };
};
