import type { SerializableObject } from 'common/types';
import { withResultAsync } from 'common/util/result';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import {
  canvasToBlob,
  canvasToImageData,
  getImageDataTransparency,
  getPrefixedId,
  previewBlob,
} from 'features/controlLayers/konva/util';
import type { GenerationMode, Rect } from 'features/controlLayers/store/types';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import type { UploadOptions } from 'services/api/endpoints/images';
import { getImageDTOSafe, uploadImage } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import stableHash from 'stable-hash';
import { assert } from 'tsafe';

/**
 * Handles compositing operations:
 * - Rasterizing and uploading the composite raster layer
 * - Rasterizing and uploading the composite inpaint mask
 * - Caclulating the generation mode (which requires the composite raster layer and inpaint mask)
 */
export class CanvasCompositorModule extends CanvasModuleBase {
  readonly type = 'compositor';
  readonly id: string;
  readonly path: string[];
  readonly log: Logger;
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;

  $isCompositing = atom(false);
  $isProcessing = atom(false);
  $isUploading = atom(false);
  $isBusy = computed(
    [this.$isCompositing, this.$isProcessing, this.$isUploading],
    (isCompositing, isProcessing, isUploading) => {
      return isCompositing || isProcessing || isUploading;
    }
  );

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('canvas_compositor');
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);
    this.log.debug('Creating compositor module');
  }

  /**
   * Gets the entity IDs of all raster layers that should be included in the composite raster layer.
   * A raster layer is included if it is enabled and has objects. The ids are sorted by draw order.
   * @returns An array of raster layer entity IDs
   */
  getCompositeRasterLayerEntityIds = (): string[] => {
    const validSortedIds = [];
    const sortedIds = this.manager.stateApi.getRasterLayersState().entities.map(({ id }) => id);
    for (const id of sortedIds) {
      const adapter = this.manager.adapters.rasterLayers.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Raster layer adapter not found');
        continue;
      }
      if (adapter.state.isEnabled && adapter.state.objects.length > 0) {
        validSortedIds.push(adapter.id);
      }
    }
    return validSortedIds;
  };

  /**
   * Gets a hash of the composite raster layer, which includes the state of all raster layers that are included in the
   * composite plus arbitrary extra data that should contribute to the hash (e.g. a rect).
   * @param extra Any extra data to include in the hash
   * @returns A hash for the composite raster layer
   */
  getCompositeRasterLayerHash = (extra: SerializableObject): string => {
    const adapterHashes: SerializableObject[] = [];

    for (const id of this.getCompositeRasterLayerEntityIds()) {
      const adapter = this.manager.adapters.rasterLayers.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Raster layer adapter not found');
        continue;
      }
      adapterHashes.push(adapter.getHashableState());
    }

    const data: SerializableObject = {
      extra,
      adapterHashes,
    };

    return stableHash(data);
  };

  /**
   * Gets a canvas element for the composite raster layer. Only the region defined by the rect is included in the canvas.
   *
   * If the hash of the composite raster layer is found in the cache, the cached canvas is returned.
   *
   * @param rect The region to include in the canvas
   * @returns A canvas element with the composite raster layer drawn on it
   */
  getCompositeRasterLayerCanvas = (rect: Rect): HTMLCanvasElement => {
    const hash = this.getCompositeRasterLayerHash({ rect });
    const cachedCanvas = this.manager.cache.canvasElementCache.get(hash);

    if (cachedCanvas) {
      this.log.trace({ rect }, 'Using cached composite raster layer canvas');
      return cachedCanvas;
    }

    this.log.trace({ rect }, 'Building composite raster layer canvas');
    this.$isCompositing.set(true);

    const canvas = document.createElement('canvas');
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    assert(ctx !== null, 'Canvas 2D context is null');

    ctx.imageSmoothingEnabled = false;

    for (const id of this.getCompositeRasterLayerEntityIds()) {
      const adapter = this.manager.adapters.rasterLayers.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Raster layer adapter not found');
        continue;
      }
      this.log.trace({ id }, 'Drawing raster layer to composite canvas');
      const adapterCanvas = adapter.getCanvas(rect);
      ctx.drawImage(adapterCanvas, 0, 0);
    }
    this.manager.cache.canvasElementCache.set(hash, canvas);
    this.$isCompositing.set(false);
    return canvas;
  };

  /**
   * Rasterizes the composite raster layer and uploads it to the server.
   *
   * If the hash of the composite raster layer is found in the cache, the cached image DTO is returned.
   *
   * @param rect The region to include in the rasterized image
   * @param options Options for uploading the image
   * @returns A promise that resolves to the uploaded image DTO
   */
  rasterizeAndUploadCompositeRasterLayer = async (
    rect: Rect,
    options: Pick<UploadOptions, 'is_intermediate' | 'metadata'>
  ): Promise<ImageDTO> => {
    this.log.trace({ rect }, 'Rasterizing composite raster layer');

    assert(rect.width > 0 && rect.height > 0, 'Unable to rasterize empty rect');

    const canvas = this.getCompositeRasterLayerCanvas(rect);

    this.$isProcessing.set(true);
    const blobResult = await withResultAsync(() => canvasToBlob(canvas));
    this.$isProcessing.set(false);

    if (blobResult.isErr()) {
      throw blobResult.error;
    }
    const blob = blobResult.value;

    if (this.manager._isDebugging) {
      previewBlob(blob, 'Composite raster layer canvas');
    }

    this.$isUploading.set(true);
    const uploadResult = await withResultAsync(() =>
      uploadImage({
        blob,
        fileName: 'composite-raster-layer.png',
        image_category: 'general',
        is_intermediate: options.is_intermediate,
        board_id: options.is_intermediate ? undefined : selectAutoAddBoardId(this.manager.store.getState()),
        metadata: options.metadata,
      })
    );
    this.$isUploading.set(false);
    if (uploadResult.isErr()) {
      throw uploadResult.error;
    }
    const imageDTO = uploadResult.value;
    return imageDTO;
  };

  /**
   * Gets the image DTO for the composite raster layer.
   *
   * If the image is found in the cache, the cached image DTO is returned.
   *
   * @param rect The region to include in the image
   * @returns A promise that resolves to the image DTO
   */
  getCompositeRasterLayerImageDTO = async (rect: Rect): Promise<ImageDTO> => {
    let imageDTO: ImageDTO | null = null;

    const hash = this.getCompositeRasterLayerHash({ rect });
    const cachedImageName = this.manager.cache.imageNameCache.get(hash);

    if (cachedImageName) {
      imageDTO = await getImageDTOSafe(cachedImageName);
      if (imageDTO) {
        this.log.trace({ rect, imageName: cachedImageName, imageDTO }, 'Using cached composite raster layer image');
        return imageDTO;
      }
    }

    imageDTO = await this.rasterizeAndUploadCompositeRasterLayer(rect, { is_intermediate: true });
    this.manager.cache.imageNameCache.set(hash, imageDTO.image_name);
    return imageDTO;
  };

  /**
   * Gets the entity IDs of all inpaint masks that should be included in the composite inpaint mask.
   * An inpaint mask is included if it is enabled and has objects. The ids are sorted by draw order.
   * @returns An array of inpaint mask entity IDs
   */
  getCompositeInpaintMaskEntityIds = (): string[] => {
    const validSortedIds = [];
    const sortedIds = this.manager.stateApi.getInpaintMasksState().entities.map(({ id }) => id);
    for (const id of sortedIds) {
      const adapter = this.manager.adapters.inpaintMasks.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Inpaint mask adapter not found');
        continue;
      }
      if (adapter.state.isEnabled && adapter.state.objects.length > 0) {
        validSortedIds.push(adapter.id);
      }
    }
    return validSortedIds;
  };

  /**
   * Gets a hash of the composite inpaint mask, which includes the state of all inpaint masks that are included in the
   * composite plus arbitrary extra data that should contribute to the hash (e.g. a rect).
   * @param extra Any extra data to include in the hash
   * @returns A hash for the composite inpaint mask
   */
  getCompositeInpaintMaskHash = (extra: SerializableObject): string => {
    const adapterHashes: SerializableObject[] = [];

    for (const id of this.getCompositeInpaintMaskEntityIds()) {
      const adapter = this.manager.adapters.inpaintMasks.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Inpaint mask adapter not found');
        continue;
      }
      adapterHashes.push(adapter.getHashableState());
    }

    const data: SerializableObject = {
      extra,
      adapterHashes,
    };

    return stableHash(data);
  };

  /**
   * Gets a canvas element for the composite inpaint mask. Only the region defined by the rect is included in the canvas.
   *
   * If the hash of the composite inpaint mask is found in the cache, the cached canvas is returned.
   *
   * @param rect The region to include in the canvas
   * @returns A canvas element with the composite inpaint mask drawn on it
   */
  getCompositeInpaintMaskCanvas = (rect: Rect): HTMLCanvasElement => {
    const hash = this.getCompositeInpaintMaskHash({ rect });
    const cachedCanvas = this.manager.cache.canvasElementCache.get(hash);

    if (cachedCanvas) {
      this.log.trace({ rect }, 'Using cached composite inpaint mask canvas');
      return cachedCanvas;
    }

    this.log.trace({ rect }, 'Building composite inpaint mask canvas');
    this.$isCompositing.set(true);
    const canvas = document.createElement('canvas');
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    assert(ctx !== null);

    ctx.imageSmoothingEnabled = false;

    for (const id of this.getCompositeInpaintMaskEntityIds()) {
      const adapter = this.manager.adapters.inpaintMasks.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Inpaint mask adapter not found');
        continue;
      }
      this.log.trace({ id }, 'Drawing inpaint mask to composite canvas');
      const adapterCanvas = adapter.getCanvas(rect);
      ctx.drawImage(adapterCanvas, 0, 0);
    }
    this.manager.cache.canvasElementCache.set(hash, canvas);
    this.$isCompositing.set(false);
    return canvas;
  };

  /**
   * Rasterizes the composite inpaint mask and uploads it to the server.
   *
   * If the hash of the composite inpaint mask is found in the cache, the cached image DTO is returned.
   *
   * @param rect The region to include in the rasterized image
   * @param saveToGallery Whether to save the image to the gallery or just return the uploaded image DTO
   * @returns A promise that resolves to the uploaded image DTO
   */
  rasterizeAndUploadCompositeInpaintMask = async (rect: Rect, saveToGallery: boolean) => {
    this.log.trace({ rect }, 'Rasterizing composite inpaint mask');

    assert(rect.width > 0 && rect.height > 0, 'Unable to rasterize empty rect');

    const canvas = this.getCompositeInpaintMaskCanvas(rect);

    this.$isProcessing.set(true);
    const blobResult = await withResultAsync(() => canvasToBlob(canvas));
    this.$isProcessing.set(false);

    if (blobResult.isErr()) {
      throw blobResult.error;
    }
    const blob = blobResult.value;

    if (this.manager._isDebugging) {
      previewBlob(blob, 'Composite inpaint mask canvas');
    }

    this.$isUploading.set(true);
    const uploadResult = await withResultAsync(() =>
      uploadImage({
        blob,
        fileName: 'composite-inpaint-mask.png',
        image_category: 'general',
        is_intermediate: !saveToGallery,
        board_id: saveToGallery ? selectAutoAddBoardId(this.manager.store.getState()) : undefined,
      })
    );
    this.$isUploading.set(false);
    if (uploadResult.isErr()) {
      throw uploadResult.error;
    }
    const imageDTO = uploadResult.value;
    return imageDTO;
  };

  /**
   * Gets the image DTO for the composite inpaint mask.
   *
   * If the image is found in the cache, the cached image DTO is returned.
   *
   * @param rect The region to include in the image
   * @returns A promise that resolves to the image DTO
   */
  getCompositeInpaintMaskImageDTO = async (rect: Rect): Promise<ImageDTO> => {
    let imageDTO: ImageDTO | null = null;

    const hash = this.getCompositeInpaintMaskHash({ rect });
    const cachedImageName = this.manager.cache.imageNameCache.get(hash);

    if (cachedImageName) {
      imageDTO = await getImageDTOSafe(cachedImageName);
      if (imageDTO) {
        this.log.trace({ rect, cachedImageName, imageDTO }, 'Using cached composite inpaint mask image');
        return imageDTO;
      }
    }

    imageDTO = await this.rasterizeAndUploadCompositeInpaintMask(rect, false);
    this.manager.cache.imageNameCache.set(hash, imageDTO.image_name);
    return imageDTO;
  };

  /**
   * Calculates the generation mode for the current canvas state. This is determined by the transparency of the
   * composite raster layer and composite inpaint mask:
   * - Composite raster layer is fully transparent -> txt2img
   * - Composite raster layer is partially transparent -> outpainting
   * - Composite raster layer is opaque & composite inpaint mask is fully transparent -> img2img
   * - Composite raster layer is opaque & composite inpaint mask is partially transparent -> inpainting
   *
   * Definitions:
   * - Fully transparent: all pixels have an alpha value of 0.
   * - Partially transparent: at least one pixel with an alpha value of 0 & at least one pixel with an alpha value
   *   greater than 0.
   * - Opaque: all pixels have an alpha value greater than 0.
   *
   * The generation mode is cached to avoid recalculating it when the canvas state has not changed.
   *
   * @returns The generation mode
   */
  getGenerationMode(): GenerationMode {
    const { rect } = this.manager.stateApi.getBbox();

    const compositeInpaintMaskHash = this.getCompositeInpaintMaskHash({ rect });
    const compositeRasterLayerHash = this.getCompositeRasterLayerHash({ rect });
    const hash = stableHash({ rect, compositeInpaintMaskHash, compositeRasterLayerHash });
    const cachedGenerationMode = this.manager.cache.generationModeCache.get(hash);

    if (cachedGenerationMode) {
      this.log.trace({ rect, cachedGenerationMode }, 'Using cached generation mode');
      return cachedGenerationMode;
    }

    const compositeInpaintMaskCanvas = this.getCompositeInpaintMaskCanvas(rect);
    this.$isProcessing.set(true);
    const compositeInpaintMaskImageData = canvasToImageData(compositeInpaintMaskCanvas);
    const compositeInpaintMaskTransparency = getImageDataTransparency(compositeInpaintMaskImageData);
    this.$isProcessing.set(false);

    const compositeRasterLayerCanvas = this.getCompositeRasterLayerCanvas(rect);
    this.$isProcessing.set(true);
    const compositeRasterLayerImageData = canvasToImageData(compositeRasterLayerCanvas);
    const compositeRasterLayerTransparency = getImageDataTransparency(compositeRasterLayerImageData);
    this.$isProcessing.set(false);

    let generationMode: GenerationMode;
    if (compositeRasterLayerTransparency === 'FULLY_TRANSPARENT') {
      // When the initial image is fully transparent, we are always doing txt2img
      generationMode = 'txt2img';
    } else if (compositeRasterLayerTransparency === 'PARTIALLY_TRANSPARENT') {
      // When the initial image is partially transparent, we are always outpainting
      generationMode = 'outpaint';
    } else if (compositeInpaintMaskTransparency === 'FULLY_TRANSPARENT') {
      // compositeLayerTransparency === 'OPAQUE'
      // When the inpaint mask is fully transparent, we are doing img2img
      generationMode = 'img2img';
    } else {
      // Else at least some of the inpaint mask is opaque, so we are inpainting
      generationMode = 'inpaint';
    }

    this.manager.cache.generationModeCache.set(hash, generationMode);
    return generationMode;
  }

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      $isCompositing: this.$isCompositing.get(),
      $isProcessing: this.$isProcessing.get(),
      $isUploading: this.$isUploading.get(),
      $isBusy: this.$isBusy.get(),
    };
  };
}
