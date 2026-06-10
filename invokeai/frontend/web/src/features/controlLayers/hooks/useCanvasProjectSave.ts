import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { parseify } from 'common/util/serialize';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { downloadBlob } from 'features/controlLayers/konva/util';
import { $currentCanvasProjectName } from 'features/controlLayers/store/currentCanvasProject';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasProjectManifest, CanvasProjectState } from 'features/controlLayers/util/canvasProjectFile';
import {
  CANVAS_PROJECT_EXTENSION,
  CANVAS_PROJECT_PREVIEW_FILENAME,
  CANVAS_PROJECT_VERSION,
  collectImageNames,
  processWithConcurrencyLimit,
} from 'features/controlLayers/util/canvasProjectFile';
import { renderCanvasProjectPreview } from 'features/controlLayers/util/canvasProjectPreview';
import { toast } from 'features/toast/toast';
import JSZip from 'jszip';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';
import {
  useReplaceCanvasProjectFileMutation,
  useUploadCanvasProjectMutation,
} from 'services/api/endpoints/canvasProjects';

const log = logger('canvas');

const sanitizeFileName = (name: string): string => {
  // Replace characters that are invalid in filenames
  return name.replace(/[<>:"/\\|?*]/g, '_').trim() || 'canvas-project';
};

/** The shape returned by buildCanvasProjectZip — bundles the ZIP itself with the metadata the
 * server-upload endpoint needs (so callers don't have to re-extract it from the manifest). */
export type BuiltCanvasProject = {
  zip: Blob;
  previewBlob: Blob | null;
  manifest: CanvasProjectManifest;
  imageCount: number;
};

export const useCanvasProjectSave = () => {
  const { t } = useTranslation();
  const store = useAppStore();
  // Read the manager out of the nanostore so this hook is safe to use outside the
  // CanvasManagerProviderGate (e.g. when the dialog is mounted via GlobalModalIsolator).
  // If the manager isn't mounted, we simply skip preview rendering.
  const canvasManager = useCanvasManagerSafe();
  const { data: appVersion } = useGetAppVersionQuery();
  const [uploadCanvasProject] = useUploadCanvasProjectMutation();
  const [replaceCanvasProjectFile] = useReplaceCanvasProjectFileMutation();

  /**
   * Builds the `.invk` ZIP in-memory plus a parallel preview WebP. Shared by both the
   * download-to-file path and the upload-to-server path so the on-disk artifacts are byte-for-byte
   * identical between the two flows.
   */
  const buildCanvasProjectZip = useCallback(
    async (name: string): Promise<BuiltCanvasProject> => {
      const state = store.getState();
      const canvasState = selectCanvasSlice(state);
      const paramsState = selectParamsSlice(state);
      const refImagesState = selectRefImagesSlice(state);
      const lorasState = selectLoRAsSlice(state);

      const projectState: CanvasProjectState = {
        rasterLayers: canvasState.rasterLayers.entities,
        controlLayers: canvasState.controlLayers.entities,
        inpaintMasks: canvasState.inpaintMasks.entities,
        regionalGuidance: canvasState.regionalGuidance.entities,
        bbox: canvasState.bbox,
        selectedEntityIdentifier: canvasState.selectedEntityIdentifier,
        bookmarkedEntityIdentifier: canvasState.bookmarkedEntityIdentifier,
      };

      const imageNames = collectImageNames(projectState, refImagesState.entities);
      const previewBlob = canvasManager
        ? await renderCanvasProjectPreview(canvasManager).catch((err) => {
            log.warn({ error: parseify(err) }, 'Failed to render canvas preview; saving without thumbnail');
            return null;
          })
        : null;

      const zip = new JSZip();

      const manifest: CanvasProjectManifest = {
        version: CANVAS_PROJECT_VERSION,
        appVersion: appVersion?.version ?? 'unknown',
        createdAt: new Date().toISOString(),
        name,
        width: canvasState.bbox.rect.width,
        height: canvasState.bbox.rect.height,
        imageCount: imageNames.size,
        hasPreview: previewBlob !== null,
      };
      zip.file('manifest.json', JSON.stringify(manifest, null, 2));

      zip.file('canvas_state.json', JSON.stringify(projectState, null, 2));
      zip.file('params.json', JSON.stringify(paramsState, null, 2));
      zip.file('ref_images.json', JSON.stringify(refImagesState.entities, null, 2));
      zip.file('loras.json', JSON.stringify(lorasState.loras, null, 2));

      if (previewBlob) {
        zip.file(CANVAS_PROJECT_PREVIEW_FILENAME, previewBlob);
      }

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

      const zipBlob = await zip.generateAsync({ type: 'blob' });
      return { zip: zipBlob, previewBlob, manifest, imageCount: imageNames.size };
    },
    [appVersion?.version, canvasManager, store]
  );

  const saveCanvasProjectAsFile = useCallback(
    async (name: string) => {
      try {
        const { zip, imageCount } = await buildCanvasProjectZip(name);
        const fileName = `${sanitizeFileName(name)}${CANVAS_PROJECT_EXTENSION}`;
        downloadBlob(zip, fileName);

        toast({
          id: 'CANVAS_PROJECT_SAVE_SUCCESS',
          title: t('controlLayers.canvasProject.saveSuccess'),
          description: t('controlLayers.canvasProject.saveSuccessDesc', { count: imageCount }),
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
    [buildCanvasProjectZip, t]
  );

  const saveCanvasProjectToServer = useCallback(
    async (name: string, boardId?: string) => {
      try {
        const { zip, previewBlob, manifest, imageCount } = await buildCanvasProjectZip(name);
        const fileName = `${sanitizeFileName(name)}${CANVAS_PROJECT_EXTENSION}`;
        const file = new File([zip], fileName, { type: 'application/zip' });

        const dto = await uploadCanvasProject({
          file,
          name,
          app_version: manifest.appVersion,
          width: manifest.width,
          height: manifest.height,
          image_count: manifest.imageCount,
          thumbnail: previewBlob ?? undefined,
          board_id: boardId,
          is_intermediate: false,
        }).unwrap();

        // Track the newly-created project as "current" so subsequent saves can update in place.
        $currentCanvasProjectName.set(dto.project_name);

        toast({
          id: 'CANVAS_PROJECT_SAVE_SUCCESS',
          title: t('controlLayers.canvasProject.saveSuccess'),
          description: t('controlLayers.canvasProject.saveSuccessDesc', { count: imageCount }),
          status: 'success',
        });
      } catch (error) {
        log.error({ error: parseify(error) }, 'Failed to save canvas project to server');
        toast({
          id: 'CANVAS_PROJECT_SAVE_ERROR',
          title: t('controlLayers.canvasProject.saveError'),
          description: String(error),
          status: 'error',
        });
      }
    },
    [buildCanvasProjectZip, t, uploadCanvasProject]
  );

  /**
   * In-place update of an existing server project. Replaces the ZIP and thumbnail but keeps the
   * project_name (UUID), board assignment, starred state and ownership. Optionally renames.
   */
  const updateCanvasProjectOnServer = useCallback(
    async (projectName: string, name: string) => {
      try {
        const { zip, previewBlob, manifest, imageCount } = await buildCanvasProjectZip(name);
        const fileName = `${sanitizeFileName(name)}${CANVAS_PROJECT_EXTENSION}`;
        const file = new File([zip], fileName, { type: 'application/zip' });

        await replaceCanvasProjectFile({
          project_name: projectName,
          file,
          name,
          app_version: manifest.appVersion,
          width: manifest.width,
          height: manifest.height,
          image_count: manifest.imageCount,
          thumbnail: previewBlob ?? undefined,
        }).unwrap();

        toast({
          id: 'CANVAS_PROJECT_SAVE_SUCCESS',
          title: t('controlLayers.canvasProject.updateSuccess'),
          description: t('controlLayers.canvasProject.saveSuccessDesc', { count: imageCount }),
          status: 'success',
        });
      } catch (error) {
        log.error({ error: parseify(error) }, 'Failed to update canvas project on server');
        toast({
          id: 'CANVAS_PROJECT_SAVE_ERROR',
          title: t('controlLayers.canvasProject.updateError'),
          description: String(error),
          status: 'error',
        });
      }
    },
    [buildCanvasProjectZip, replaceCanvasProjectFile, t]
  );

  return { saveCanvasProjectAsFile, saveCanvasProjectToServer, updateCanvasProjectOnServer };
};
