import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import { parseify } from 'common/util/serialize';
import { canvasProjectRecalled } from 'features/controlLayers/store/canvasSlice';
import { loraAllDeleted, loraRecalled } from 'features/controlLayers/store/lorasSlice';
import { paramsRecalled } from 'features/controlLayers/store/paramsSlice';
import { refImagesRecalled } from 'features/controlLayers/store/refImagesSlice';
import type { LoRA, ParamsState, RefImageState } from 'features/controlLayers/store/types';
import type { CanvasProjectState } from 'features/controlLayers/util/canvasProjectFile';
import {
  checkExistingImages,
  collectImageNames,
  parseManifest,
  processWithConcurrencyLimit,
  remapCanvasState,
  remapRefImages,
} from 'features/controlLayers/util/canvasProjectFile';
import { toast } from 'features/toast/toast';
import JSZip from 'jszip';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { uploadImage } from 'services/api/endpoints/images';

const log = logger('canvas');

export const useCanvasProjectLoad = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const loadCanvasProject = useCallback(
    async (file: File) => {
      try {
        const zip = await JSZip.loadAsync(file);

        // Validate manifest
        const manifestFile = zip.file('manifest.json');
        if (!manifestFile) {
          throw new Error('Invalid project file: missing manifest.json');
        }
        const manifestData = JSON.parse(await manifestFile.async('string'));
        parseManifest(manifestData);

        // Read state files
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

        // Collect all image names referenced in the state
        const imageNames = collectImageNames(canvasState, refImages);

        // Check which images already exist on the server
        const { missing } = await checkExistingImages(imageNames);

        // Upload missing images from the ZIP
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

              // Map old name to new name (only if different)
              if (imageDTO.image_name !== imageName) {
                imageNameMapping.set(imageName, imageDTO.image_name);
              }
            } catch (error) {
              log.warn({ error: parseify(error) }, `Failed to upload image ${imageName}`);
            }
          });
        }

        // Remap image names in state objects
        const remappedCanvasState = remapCanvasState(canvasState, imageNameMapping);
        const remappedRefImages = remapRefImages(refImages, imageNameMapping);

        // Dispatch state restoration
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

        // Restore reference images
        dispatch(refImagesRecalled({ entities: remappedRefImages, replace: true }));

        // Restore generation parameters
        if (projectParams) {
          dispatch(paramsRecalled(projectParams));
        }

        // Restore LoRAs (always clear, even if project has none)
        dispatch(loraAllDeleted());
        for (const lora of loras) {
          dispatch(loraRecalled({ lora }));
        }

        toast({
          id: 'CANVAS_PROJECT_LOAD_SUCCESS',
          title: t('controlLayers.canvasProject.loadSuccess'),
          description: t('controlLayers.canvasProject.loadSuccessDesc'),
          status: 'success',
        });
      } catch (error) {
        log.error({ error: parseify(error) }, 'Failed to load canvas project');
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

  return { loadCanvasProject };
};
