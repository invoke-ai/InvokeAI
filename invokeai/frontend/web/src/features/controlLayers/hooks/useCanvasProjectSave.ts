import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { parseify } from 'common/util/serialize';
import { downloadBlob } from 'features/controlLayers/konva/util';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasProjectManifest, CanvasProjectState } from 'features/controlLayers/util/canvasProjectFile';
import {
  CANVAS_PROJECT_EXTENSION,
  CANVAS_PROJECT_VERSION,
  collectImageNames,
  processWithConcurrencyLimit,
} from 'features/controlLayers/util/canvasProjectFile';
import { toast } from 'features/toast/toast';
import JSZip from 'jszip';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

const log = logger('canvas');

const sanitizeFileName = (name: string): string => {
  // Replace characters that are invalid in filenames
  return name.replace(/[<>:"/\\|?*]/g, '_').trim() || 'canvas-project';
};

export const useCanvasProjectSave = () => {
  const { t } = useTranslation();
  const store = useAppStore();
  const { data: appVersion } = useGetAppVersionQuery();

  const saveCanvasProject = useCallback(
    async (name: string) => {
      try {
        const state = store.getState();
        const canvasState = selectCanvasSlice(state);
        const paramsState = selectParamsSlice(state);
        const refImagesState = selectRefImagesSlice(state);
        const lorasState = selectLoRAsSlice(state);

        // Build the canvas project state
        const projectState: CanvasProjectState = {
          rasterLayers: canvasState.rasterLayers.entities,
          controlLayers: canvasState.controlLayers.entities,
          inpaintMasks: canvasState.inpaintMasks.entities,
          regionalGuidance: canvasState.regionalGuidance.entities,
          bbox: canvasState.bbox,
          selectedEntityIdentifier: canvasState.selectedEntityIdentifier,
          bookmarkedEntityIdentifier: canvasState.bookmarkedEntityIdentifier,
        };

        // Collect all image names referenced in the state
        const imageNames = collectImageNames(projectState, refImagesState.entities);

        // Build ZIP
        const zip = new JSZip();

        // Add manifest
        const manifest: CanvasProjectManifest = {
          version: CANVAS_PROJECT_VERSION,
          appVersion: appVersion?.version ?? 'unknown',
          createdAt: new Date().toISOString(),
          name,
        };
        zip.file('manifest.json', JSON.stringify(manifest, null, 2));

        // Add state files
        zip.file('canvas_state.json', JSON.stringify(projectState, null, 2));
        zip.file('params.json', JSON.stringify(paramsState, null, 2));
        zip.file('ref_images.json', JSON.stringify(refImagesState.entities, null, 2));
        zip.file('loras.json', JSON.stringify(lorasState.loras, null, 2));

        // Fetch and add images
        const imagesFolder = zip.folder('images')!;
        await processWithConcurrencyLimit(Array.from(imageNames), async (imageName) => {
          try {
            const response = await fetch(`/api/v1/images/i/${imageName}/full`);
            if (!response.ok) {
              log.warn(`Failed to fetch image ${imageName}: ${response.status}`);
              return;
            }
            const blob = await response.blob();
            imagesFolder.file(imageName, blob);
          } catch (error) {
            log.warn({ error: parseify(error) }, `Failed to fetch image ${imageName}`);
          }
        });

        // Generate ZIP blob and trigger download
        const blob = await zip.generateAsync({ type: 'blob' });
        const fileName = `${sanitizeFileName(name)}${CANVAS_PROJECT_EXTENSION}`;
        downloadBlob(blob, fileName);

        toast({
          id: 'CANVAS_PROJECT_SAVE_SUCCESS',
          title: t('controlLayers.canvasProject.saveSuccess'),
          description: t('controlLayers.canvasProject.saveSuccessDesc', { count: imageNames.size }),
          status: 'success',
        });
      } catch (error) {
        log.error({ error: parseify(error) }, 'Failed to save canvas project');
        toast({
          id: 'CANVAS_PROJECT_SAVE_ERROR',
          title: t('controlLayers.canvasProject.saveError'),
          description: String(error),
          status: 'error',
        });
      }
    },
    [appVersion?.version, store, t]
  );

  return { saveCanvasProject };
};
