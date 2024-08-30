import type { SerializableObject } from 'common/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleABC } from 'features/controlLayers/konva/CanvasModuleABC';
import {
  canvasToBlob,
  canvasToImageData,
  getImageDataTransparency,
  getPrefixedId,
  previewBlob,
} from 'features/controlLayers/konva/util';
import type { GenerationMode, Rect } from 'features/controlLayers/store/types';
import type { Logger } from 'roarr';
import { getImageDTO, uploadImage } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import stableHash from 'stable-hash';
import { assert } from 'tsafe';

export class CanvasCompositorModule extends CanvasModuleABC {
  readonly type = 'compositor';

  id: string;
  path: string[];
  log: Logger;
  manager: CanvasManager;
  subscriptions = new Set<() => void>();

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('canvas_compositor');
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug('Creating compositor module');
  }

  getCompositeRasterLayerEntityIds = (): string[] => {
    const ids = [];
    for (const adapter of this.manager.adapters.rasterLayers.values()) {
      if (adapter.state.isEnabled && adapter.renderer.hasObjects()) {
        ids.push(adapter.id);
      }
    }
    return ids;
  };

  getCompositeInpaintMaskEntityIds = (): string[] => {
    const ids = [];
    for (const adapter of this.manager.adapters.inpaintMasks.values()) {
      if (adapter.state.isEnabled && adapter.renderer.hasObjects()) {
        ids.push(adapter.id);
      }
    }
    return ids;
  };

  getCompositeRasterLayerCanvas = (rect: Rect): HTMLCanvasElement => {
    const hash = this.getCompositeRasterLayerHash({ rect });
    const cachedCanvas = this.manager.cache.canvasElementCache.get(hash);

    if (cachedCanvas) {
      this.log.trace({ rect }, 'Using cached composite inpaint mask canvas');
      return cachedCanvas;
    }

    this.log.trace({ rect }, 'Building composite raster layer canvas');

    const canvas = document.createElement('canvas');
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    assert(ctx !== null, 'Canvas 2D context is null');

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
    return canvas;
  };

  getCompositeInpaintMaskCanvas = (rect: Rect): HTMLCanvasElement => {
    const hash = this.getCompositeInpaintMaskHash({ rect });
    const cachedCanvas = this.manager.cache.canvasElementCache.get(hash);

    if (cachedCanvas) {
      this.log.trace({ rect }, 'Using cached composite inpaint mask canvas');
      return cachedCanvas;
    }

    this.log.trace({ rect }, 'Building composite inpaint mask canvas');

    const canvas = document.createElement('canvas');
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    assert(ctx !== null);

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
    return canvas;
  };

  getCompositeRasterLayerHash = (extra: SerializableObject): string => {
    const data: Record<string, SerializableObject> = {
      extra,
    };
    for (const id of this.getCompositeRasterLayerEntityIds()) {
      const adapter = this.manager.adapters.rasterLayers.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Raster layer adapter not found');
        continue;
      }
      data[id] = adapter.getHashableState();
    }
    return stableHash(data);
  };

  getCompositeInpaintMaskHash = (extra: SerializableObject): string => {
    const data: Record<string, SerializableObject> = {
      extra,
    };
    for (const id of this.getCompositeInpaintMaskEntityIds()) {
      const adapter = this.manager.adapters.inpaintMasks.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Inpaint mask adapter not found');
        continue;
      }
      data[id] = adapter.getHashableState();
    }
    return stableHash(data);
  };

  rasterizeAndUploadCompositeRasterLayer = async (rect: Rect, saveToGallery: boolean) => {
    this.log.trace({ rect }, 'Rasterizing composite raster layer');

    const canvas = this.getCompositeRasterLayerCanvas(rect);
    const blob = await canvasToBlob(canvas);

    if (this.manager._isDebugging) {
      previewBlob(blob, 'Composite raster layer canvas');
    }

    return uploadImage(blob, 'composite-raster-layer.png', 'general', !saveToGallery);
  };

  getCompositeRasterLayerImageDTO = async (rect: Rect): Promise<ImageDTO> => {
    let imageDTO: ImageDTO | null = null;

    const hash = this.getCompositeRasterLayerHash({ rect });
    const cachedImageName = this.manager.cache.imageNameCache.get(hash);

    if (cachedImageName) {
      imageDTO = await getImageDTO(cachedImageName);
      if (imageDTO) {
        this.log.trace({ rect, imageName: cachedImageName, imageDTO }, 'Using cached composite raster layer image');
        return imageDTO;
      }
    }

    imageDTO = await this.rasterizeAndUploadCompositeRasterLayer(rect, false);
    this.manager.cache.imageNameCache.set(hash, imageDTO.image_name);
    return imageDTO;
  };

  rasterizeAndUploadCompositeInpaintMask = async (rect: Rect, saveToGallery: boolean) => {
    this.log.trace({ rect }, 'Rasterizing composite inpaint mask');

    const canvas = this.getCompositeInpaintMaskCanvas(rect);
    const blob = await canvasToBlob(canvas);
    if (this.manager._isDebugging) {
      previewBlob(blob, 'Composite inpaint mask canvas');
    }

    return uploadImage(blob, 'composite-inpaint-mask.png', 'general', !saveToGallery);
  };

  getCompositeInpaintMaskImageDTO = async (rect: Rect): Promise<ImageDTO> => {
    let imageDTO: ImageDTO | null = null;

    const hash = this.getCompositeInpaintMaskHash({ rect });
    const cachedImageName = this.manager.cache.imageNameCache.get(hash);

    if (cachedImageName) {
      imageDTO = await getImageDTO(cachedImageName);
      if (imageDTO) {
        this.log.trace({ rect, cachedImageName, imageDTO }, 'Using cached composite inpaint mask image');
        return imageDTO;
      }
    }

    imageDTO = await this.rasterizeAndUploadCompositeInpaintMask(rect, false);
    this.manager.cache.imageNameCache.set(hash, imageDTO.image_name);
    return imageDTO;
  };

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
    const compositeInpaintMaskImageData = canvasToImageData(compositeInpaintMaskCanvas);
    const compositeInpaintMaskTransparency = getImageDataTransparency(compositeInpaintMaskImageData);

    const compositeRasterLayerCanvas = this.getCompositeRasterLayerCanvas(rect);
    const compositeRasterLayerImageData = canvasToImageData(compositeRasterLayerCanvas);
    const compositeRasterLayerTransparency = getImageDataTransparency(compositeRasterLayerImageData);

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
    };
  };

  destroy = () => {
    this.log.trace('Destroying compositor module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
  };

  getLoggingContext = () => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
