import { logger } from 'app/logging/logger';
import { getStore } from 'app/store/nanostores/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { parseify } from 'common/util/serialize';
import { canvasProjectRecalled } from 'features/controlLayers/store/canvasSlice';
import { $currentCanvasProjectName } from 'features/controlLayers/store/currentCanvasProject';
import { loraAllDeleted, loraRecalled } from 'features/controlLayers/store/lorasSlice';
import { paramsRecalled } from 'features/controlLayers/store/paramsSlice';
import { refImagesRecalled } from 'features/controlLayers/store/refImagesSlice';
import type { LoRA, ParamsState, RefImageState } from 'features/controlLayers/store/types';
import type { AnyCanvasProjectManifest, CanvasProjectState } from 'features/controlLayers/util/canvasProjectFile';
import {
  CANVAS_PROJECT_PREVIEW_FILENAME,
  CANVAS_PROJECT_VERSION,
  checkExistingImages,
  collectImageNames,
  isV2Manifest,
  parseManifest,
  processWithConcurrencyLimit,
  remapCanvasState,
  remapRefImages,
} from 'features/controlLayers/util/canvasProjectFile';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import JSZip from 'jszip';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { canvasProjectsApi, fetchCanvasProjectZip } from 'services/api/endpoints/canvasProjects';
import { uploadImage } from 'services/api/endpoints/images';

const log = logger('canvas');

type ParseResult = {
  manifest: AnyCanvasProjectManifest;
  remappedCanvasState: CanvasProjectState;
  remappedRefImages: RefImageState[];
  imageNameMapping: Map<string, string>;
  remappedImageCount: number;
};

/**
 * Parses an in-memory `.invk` ZIP and dispatches the canvas restore actions.
 *
 * Returns metadata about the parse so callers can decide whether to trigger an automatic
 * server-side re-save (for projects loaded from the server when remapping happened).
 */
const parseAndApplyCanvasProjectZip = async (
  blob: Blob,
  dispatch: ReturnType<typeof useAppDispatch>
): Promise<ParseResult> => {
  const zip = await JSZip.loadAsync(blob);

  const manifestFile = zip.file('manifest.json');
  if (!manifestFile) {
    throw new Error('Invalid project file: missing manifest.json');
  }
  const manifestData = JSON.parse(await manifestFile.async('string'));
  const manifest = parseManifest(manifestData);

  const canvasStateFile = zip.file('canvas_state.json');
  if (!canvasStateFile) {
    throw new Error('Invalid project file: missing canvas_state.json');
  }
  const canvasState: CanvasProjectState = JSON.parse(await canvasStateFile.async('string'));

  const paramsFile = zip.file('params.json');
  let projectParams: ParamsState | null = null;
  if (paramsFile) {
    projectParams = JSON.parse(await paramsFile.async('string'));
  }

  const refImagesFile = zip.file('ref_images.json');
  let refImages: RefImageState[] = [];
  if (refImagesFile) {
    refImages = JSON.parse(await refImagesFile.async('string'));
  }

  const lorasFile = zip.file('loras.json');
  let loras: LoRA[] = [];
  if (lorasFile) {
    loras = JSON.parse(await lorasFile.async('string'));
  }

  const imageNames = collectImageNames(canvasState, refImages);
  const { missing } = await checkExistingImages(imageNames);

  const imageNameMapping = new Map<string, string>();
  const imagesFolder = zip.folder('images');

  if (imagesFolder && missing.size > 0) {
    await processWithConcurrencyLimit(Array.from(missing), async (imageName) => {
      const imageFile = imagesFolder.file(imageName);
      if (!imageFile) {
        log.warn(`Image ${imageName} referenced but not found in ZIP`);
        return;
      }

      try {
        const blob = await imageFile.async('blob');
        const uploadFile = new File([blob], imageName, { type: 'image/png' });
        const imageDTO = await uploadImage({
          file: uploadFile,
          image_category: 'general',
          is_intermediate: false,
          silent: true,
        });

        if (imageDTO.image_name !== imageName) {
          imageNameMapping.set(imageName, imageDTO.image_name);
        }
      } catch (error) {
        log.warn({ error: parseify(error) }, `Failed to upload image ${imageName}`);
      }
    });
  }

  const remappedCanvasState = remapCanvasState(canvasState, imageNameMapping);
  const remappedRefImages = remapRefImages(refImages, imageNameMapping);

  dispatch(
    canvasProjectRecalled({
      rasterLayers: remappedCanvasState.rasterLayers,
      controlLayers: remappedCanvasState.controlLayers,
      inpaintMasks: remappedCanvasState.inpaintMasks,
      regionalGuidance: remappedCanvasState.regionalGuidance,
      bbox: remappedCanvasState.bbox,
      selectedEntityIdentifier: remappedCanvasState.selectedEntityIdentifier,
      bookmarkedEntityIdentifier: remappedCanvasState.bookmarkedEntityIdentifier,
    })
  );

  dispatch(refImagesRecalled({ entities: remappedRefImages, replace: true }));

  if (projectParams) {
    dispatch(paramsRecalled(projectParams));
  }

  // Always clear LoRAs first, even when the project has none — otherwise an in-place load would
  // accumulate LoRAs across sessions.
  dispatch(loraAllDeleted());
  for (const lora of loras) {
    dispatch(loraRecalled({ lora }));
  }

  // Re-collect names after remapping so the auto-resave path can report the correct count.
  const remappedImageCount = collectImageNames(remappedCanvasState, remappedRefImages).size;

  return { manifest, remappedCanvasState, remappedRefImages, imageNameMapping, remappedImageCount };
};

/**
 * Rebuilds a `.invk` ZIP from the original ZIP plus the remapped state. Image bytes are renamed
 * to the new server-side names but their content is preserved bit-for-bit. The preview, params,
 * and loras files are passed through unchanged.
 *
 * This is what makes the auto-resave path idempotent: after we replace the server-side file with
 * a rebuilt ZIP, the next load will find every referenced image already present on the server
 * and skip the re-upload entirely.
 */
const rebuildZipWithRemapping = async (
  originalBlob: Blob,
  remappedCanvasState: CanvasProjectState,
  remappedRefImages: RefImageState[],
  manifest: AnyCanvasProjectManifest,
  remappedImageCount: number,
  imageNameMapping: Map<string, string>
): Promise<Blob> => {
  const original = await JSZip.loadAsync(originalBlob);
  const next = new JSZip();

  // v2 manifest with refreshed image_count.
  const hasPreview = isV2Manifest(manifest)
    ? manifest.hasPreview
    : original.file(CANVAS_PROJECT_PREVIEW_FILENAME) !== null;
  const width = isV2Manifest(manifest) ? manifest.width : remappedCanvasState.bbox.rect.width;
  const height = isV2Manifest(manifest) ? manifest.height : remappedCanvasState.bbox.rect.height;

  const newManifest = {
    version: CANVAS_PROJECT_VERSION,
    appVersion: manifest.appVersion,
    createdAt: manifest.createdAt,
    name: manifest.name,
    width,
    height,
    imageCount: remappedImageCount,
    hasPreview,
  };
  next.file('manifest.json', JSON.stringify(newManifest, null, 2));
  next.file('canvas_state.json', JSON.stringify(remappedCanvasState, null, 2));
  next.file('ref_images.json', JSON.stringify(remappedRefImages, null, 2));

  // Pass-through files that aren't affected by image-name remapping.
  for (const passthroughName of ['params.json', 'loras.json', CANVAS_PROJECT_PREVIEW_FILENAME]) {
    const f = original.file(passthroughName);
    if (f) {
      next.file(passthroughName, await f.async('blob'));
    }
  }

  // Copy images, renaming any that were remapped.
  const imagesFolder = original.folder('images');
  if (imagesFolder) {
    const entries: { originalName: string; relativePath: string }[] = [];
    imagesFolder.forEach((relativePath, file) => {
      if (file.dir) {
        return;
      }
      entries.push({ originalName: relativePath, relativePath: file.name });
    });

    const targetFolder = next.folder('images')!;
    for (const { originalName, relativePath } of entries) {
      const file = original.file(relativePath);
      if (!file) {
        continue;
      }
      const newName = imageNameMapping.get(originalName) ?? originalName;
      targetFolder.file(newName, await file.async('blob'));
    }
  }

  return await next.generateAsync({ type: 'blob' });
};

/**
 * Pulls the bytes of the existing preview thumbnail out of the original ZIP so we can include
 * it in the auto-resave (the server's existing thumbnail file gets overwritten on replace, and
 * we don't want to lose the preview just because we re-saved without a canvas manager available
 * to render a fresh one).
 */
const extractPreviewBlob = async (originalBlob: Blob): Promise<Blob | null> => {
  const zip = await JSZip.loadAsync(originalBlob);
  const f = zip.file(CANVAS_PROJECT_PREVIEW_FILENAME);
  return f ? await f.async('blob') : null;
};

export const useCanvasProjectLoad = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const loadCanvasProjectFromFile = useCallback(
    async (file: File) => {
      try {
        await parseAndApplyCanvasProjectZip(file, dispatch);
        // File-loads aren't tracked as "currently loaded server project" — there's nothing to
        // update on the server. Clear any stale tracking from a previous server-load.
        $currentCanvasProjectName.set(null);
        navigationApi.switchToTab('canvas');
        toast({
          id: 'CANVAS_PROJECT_LOAD_SUCCESS',
          title: t('controlLayers.canvasProject.loadSuccess'),
          description: t('controlLayers.canvasProject.loadSuccessDesc'),
          status: 'success',
        });
      } catch (error) {
        log.error({ error: parseify(error) }, 'Failed to load canvas project from file');
        toast({
          id: 'CANVAS_PROJECT_LOAD_ERROR',
          title: t('controlLayers.canvasProject.loadError'),
          description: String(error),
          status: 'error',
        });
      }
    },
    [dispatch, t]
  );

  const loadCanvasProjectFromServer = useCallback(
    async (projectName: string) => {
      try {
        const blob = await fetchCanvasProjectZip(projectName);
        const result = await parseAndApplyCanvasProjectZip(blob, dispatch);
        $currentCanvasProjectName.set(projectName);
        navigationApi.switchToTab('canvas');

        // Auto-resave: if remapping happened, the server-side ZIP still references the missing
        // image names. Push a rebuilt ZIP back so subsequent loads don't re-upload the same bytes.
        if (result.imageNameMapping.size > 0) {
          try {
            const rebuilt = await rebuildZipWithRemapping(
              blob,
              result.remappedCanvasState,
              result.remappedRefImages,
              result.manifest,
              result.remappedImageCount,
              result.imageNameMapping
            );
            const previewBlob = await extractPreviewBlob(blob);
            const rebuiltFile = new File([rebuilt], `${projectName}.invk`, { type: 'application/zip' });
            await getStore()
              .dispatch(
                canvasProjectsApi.endpoints.replaceCanvasProjectFile.initiate(
                  {
                    project_name: projectName,
                    file: rebuiltFile,
                    app_version: result.manifest.appVersion,
                    width: isV2Manifest(result.manifest)
                      ? result.manifest.width
                      : result.remappedCanvasState.bbox.rect.width,
                    height: isV2Manifest(result.manifest)
                      ? result.manifest.height
                      : result.remappedCanvasState.bbox.rect.height,
                    image_count: result.remappedImageCount,
                    thumbnail: previewBlob ?? undefined,
                  },
                  { track: false }
                )
              )
              .unwrap();
            log.info(`Auto-resaved canvas project ${projectName} after ${result.imageNameMapping.size} image remap(s)`);
          } catch (replaceErr) {
            // Soft-fail: the load itself succeeded, the user has a working canvas. Surfacing this
            // as a hard error would be noisy when the underlying restore is fine.
            log.warn({ error: parseify(replaceErr) }, 'Auto-resave after remapping failed');
          }
        }

        toast({
          id: 'CANVAS_PROJECT_LOAD_SUCCESS',
          title: t('controlLayers.canvasProject.loadSuccess'),
          description: t('controlLayers.canvasProject.loadSuccessDesc'),
          status: 'success',
        });
      } catch (error) {
        log.error({ error: parseify(error) }, 'Failed to load canvas project from server');
        toast({
          id: 'CANVAS_PROJECT_LOAD_ERROR',
          title: t('controlLayers.canvasProject.loadError'),
          description: String(error),
          status: 'error',
        });
      }
    },
    [dispatch, t]
  );

  return { loadCanvasProjectFromFile, loadCanvasProjectFromServer };
};
