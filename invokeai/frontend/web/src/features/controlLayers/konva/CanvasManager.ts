import type { AppSocket } from 'app/hooks/useSocketIO';
import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import type { SerializableObject } from 'common/types';
import { CanvasCacheModule } from 'features/controlLayers/konva/CanvasCacheModule';
import { CanvasFilter } from 'features/controlLayers/konva/CanvasFilter';
import { CanvasRenderingModule } from 'features/controlLayers/konva/CanvasRenderingModule';
import { CanvasStageModule } from 'features/controlLayers/konva/CanvasStageModule';
import { CanvasWorkerModule } from 'features/controlLayers/konva/CanvasWorkerModule.js';
import {
  canvasToBlob,
  canvasToImageData,
  getImageDataTransparency,
  getPrefixedId,
  previewBlob,
} from 'features/controlLayers/konva/util';
import type { GenerationMode, Rect } from 'features/controlLayers/store/types';
import type Konva from 'konva';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { getImageDTO, uploadImage } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import stableHash from 'stable-hash';
import { assert } from 'tsafe';

import { CanvasBackground } from './CanvasBackground';
import type { CanvasLayerAdapter } from './CanvasLayerAdapter';
import type { CanvasMaskAdapter } from './CanvasMaskAdapter';
import { CanvasPreview } from './CanvasPreview';
import { CanvasStateApi } from './CanvasStateApi';
import { setStageEventHandlers } from './events';

export const $canvasManager = atom<CanvasManager | null>(null);
const TYPE = 'manager';

export class CanvasManager {
  readonly type = TYPE;

  id: string;
  path: string[];

  store: AppStore;
  socket: AppSocket;

  rasterLayerAdapters: Map<string, CanvasLayerAdapter> = new Map();
  controlLayerAdapters: Map<string, CanvasLayerAdapter> = new Map();
  regionalGuidanceAdapters: Map<string, CanvasMaskAdapter> = new Map();
  inpaintMaskAdapters: Map<string, CanvasMaskAdapter> = new Map();

  stateApi: CanvasStateApi;
  preview: CanvasPreview;
  background: CanvasBackground;
  filter: CanvasFilter;
  stage: CanvasStageModule;
  worker: CanvasWorkerModule;
  cache: CanvasCacheModule;
  renderer: CanvasRenderingModule;

  _isDebugging: boolean = false;

  constructor(stage: Konva.Stage, container: HTMLDivElement, store: AppStore, socket: AppSocket) {
    this.id = getPrefixedId(this.type);
    this.path = [this.id];
    this.store = store;
    this.socket = socket;
    this.stateApi = new CanvasStateApi(this.store, this);

    this.stage = new CanvasStageModule(stage, container, this);
    this.worker = new CanvasWorkerModule(this);
    this.cache = new CanvasCacheModule(this);
    this.renderer = new CanvasRenderingModule(this);
    this.preview = new CanvasPreview(this);
    this.stage.addLayer(this.preview.getLayer());

    this.background = new CanvasBackground(this);
    this.stage.addLayer(this.background.konva.layer);

    this.filter = new CanvasFilter(this);
  }

  log = logger('canvas').child((message) => {
    return {
      ...message,
      context: {
        ...this.getLoggingContext(),
        ...message.context,
      },
    };
  });

  enableDebugging() {
    this._isDebugging = true;
    this.logDebugInfo();
  }

  disableDebugging() {
    this._isDebugging = false;
  }

  getTransformingLayer = (): CanvasLayerAdapter | CanvasMaskAdapter | null => {
    const transformingEntity = this.stateApi.$transformingEntity.get();
    if (!transformingEntity) {
      return null;
    }

    const { id, type } = transformingEntity;

    if (type === 'raster_layer') {
      return this.rasterLayerAdapters.get(id) ?? null;
    } else if (type === 'control_layer') {
      return this.controlLayerAdapters.get(id) ?? null;
    } else if (type === 'inpaint_mask') {
      return this.inpaintMaskAdapters.get(id) ?? null;
    } else if (type === 'regional_guidance') {
      return this.regionalGuidanceAdapters.get(id) ?? null;
    }

    return null;
  };

  getIsTransforming() {
    return Boolean(this.stateApi.$transformingEntity.get());
  }

  startTransform() {
    if (this.getIsTransforming()) {
      return;
    }
    const entity = this.stateApi.getSelectedEntity();
    if (!entity) {
      this.log.warn('No entity selected to transform');
      return;
    }
    // TODO(psyche): Support other entity types
    entity.adapter.transformer.startTransform();
    this.stateApi.$transformingEntity.set({ id: entity.id, type: entity.type });
  }

  async applyTransform() {
    const layer = this.getTransformingLayer();
    if (layer) {
      await layer.transformer.applyTransform();
    }
    this.stateApi.$transformingEntity.set(null);
  }

  cancelTransform() {
    const layer = this.getTransformingLayer();
    if (layer) {
      layer.transformer.stopTransform();
    }
    this.stateApi.$transformingEntity.set(null);
  }

  initialize = () => {
    this.log.debug('Initializing canvas manager');

    // These atoms require the canvas manager to be set up before we can provide their initial values
    this.stateApi.$transformingEntity.set(null);
    this.stateApi.$toolState.set(this.stateApi.getToolState());
    this.stateApi.$selectedEntityIdentifier.set(this.stateApi.getState().selectedEntityIdentifier);
    this.stateApi.$currentFill.set(this.stateApi.getCurrentFill());
    this.stateApi.$selectedEntity.set(this.stateApi.getSelectedEntity());

    const cleanupEventHandlers = setStageEventHandlers(this);
    const cleanupStage = this.stage.initialize();
    const cleanupStore = this.store.subscribe(this.renderer.render);

    return () => {
      this.log.debug('Cleaning up canvas manager');
      const allAdapters = [
        ...this.rasterLayerAdapters.values(),
        ...this.controlLayerAdapters.values(),
        ...this.inpaintMaskAdapters.values(),
        ...this.regionalGuidanceAdapters.values(),
      ];
      for (const adapter of allAdapters) {
        adapter.destroy();
      }
      this.background.destroy();
      this.preview.destroy();
      cleanupStore();
      cleanupEventHandlers();
      cleanupStage();
    };
  };

  getCompositeRasterLayerEntityIds = (): string[] => {
    const ids = [];
    for (const adapter of this.rasterLayerAdapters.values()) {
      if (adapter.state.isEnabled && adapter.renderer.hasObjects()) {
        ids.push(adapter.id);
      }
    }
    return ids;
  };

  getCompositeInpaintMaskEntityIds = (): string[] => {
    const ids = [];
    for (const adapter of this.inpaintMaskAdapters.values()) {
      if (adapter.state.isEnabled && adapter.renderer.hasObjects()) {
        ids.push(adapter.id);
      }
    }
    return ids;
  };

  getCompositeRasterLayerCanvas = (rect: Rect): HTMLCanvasElement => {
    const hash = this.getCompositeRasterLayerHash({ rect });
    const cachedCanvas = this.cache.canvasElementCache.get(hash);

    if (cachedCanvas) {
      this.log.trace({ rect }, 'Using cached composite inpaint mask canvas');
      return cachedCanvas;
    }

    this.log.trace({ rect }, 'Building composite raster layer canvas');

    const canvas = document.createElement('canvas');
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    assert(ctx !== null);

    for (const id of this.getCompositeRasterLayerEntityIds()) {
      const adapter = this.rasterLayerAdapters.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Raster layer adapter not found');
        continue;
      }
      this.log.trace({ id }, 'Drawing raster layer to composite canvas');
      const adapterCanvas = adapter.getCanvas(rect);
      ctx.drawImage(adapterCanvas, 0, 0);
    }
    this.cache.canvasElementCache.set(hash, canvas);
    return canvas;
  };

  getCompositeInpaintMaskCanvas = (rect: Rect): HTMLCanvasElement => {
    const hash = this.getCompositeInpaintMaskHash({ rect });
    const cachedCanvas = this.cache.canvasElementCache.get(hash);

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
      const adapter = this.inpaintMaskAdapters.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Inpaint mask adapter not found');
        continue;
      }
      this.log.trace({ id }, 'Drawing inpaint mask to composite canvas');
      const adapterCanvas = adapter.getCanvas(rect);
      ctx.drawImage(adapterCanvas, 0, 0);
    }
    this.cache.canvasElementCache.set(hash, canvas);
    return canvas;
  };

  getCompositeRasterLayerHash = (extra: SerializableObject): string => {
    const data: Record<string, SerializableObject> = {
      extra,
    };
    for (const id of this.getCompositeRasterLayerEntityIds()) {
      const adapter = this.rasterLayerAdapters.get(id);
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
      const adapter = this.inpaintMaskAdapters.get(id);
      if (!adapter) {
        this.log.warn({ id }, 'Inpaint mask adapter not found');
        continue;
      }
      data[id] = adapter.getHashableState();
    }
    return stableHash(data);
  };

  getCompositeRasterLayerImageDTO = async (rect: Rect): Promise<ImageDTO> => {
    let imageDTO: ImageDTO | null = null;

    const hash = this.getCompositeRasterLayerHash({ rect });
    const cachedImageName = this.cache.imageNameCache.get(hash);

    if (cachedImageName) {
      imageDTO = await getImageDTO(cachedImageName);
      if (imageDTO) {
        this.log.trace({ rect, imageName: cachedImageName, imageDTO }, 'Using cached composite raster layer image');
        return imageDTO;
      }
    }

    this.log.trace({ rect }, 'Rasterizing composite raster layer');

    const canvas = this.getCompositeRasterLayerCanvas(rect);
    const blob = await canvasToBlob(canvas);
    if (this._isDebugging) {
      previewBlob(blob, 'Composite raster layer canvas');
    }

    imageDTO = await uploadImage(blob, 'composite-raster-layer.png', 'general', true);
    this.cache.imageNameCache.set(hash, imageDTO.image_name);
    return imageDTO;
  };

  getCompositeInpaintMaskImageDTO = async (rect: Rect): Promise<ImageDTO> => {
    let imageDTO: ImageDTO | null = null;

    const hash = this.getCompositeInpaintMaskHash({ rect });
    const cachedImageName = this.cache.imageNameCache.get(hash);

    if (cachedImageName) {
      imageDTO = await getImageDTO(cachedImageName);
      if (imageDTO) {
        this.log.trace({ rect, cachedImageName, imageDTO }, 'Using cached composite inpaint mask image');
        return imageDTO;
      }
    }

    this.log.trace({ rect }, 'Rasterizing composite inpaint mask');

    const canvas = this.getCompositeInpaintMaskCanvas(rect);
    const blob = await canvasToBlob(canvas);
    if (this._isDebugging) {
      previewBlob(blob, 'Composite inpaint mask canvas');
    }

    imageDTO = await uploadImage(blob, 'composite-inpaint-mask.png', 'general', true);
    this.cache.imageNameCache.set(hash, imageDTO.image_name);
    return imageDTO;
  };

  getGenerationMode(): GenerationMode {
    const { rect } = this.stateApi.getBbox();

    const compositeInpaintMaskHash = this.getCompositeInpaintMaskHash({ rect });
    const compositeRasterLayerHash = this.getCompositeRasterLayerHash({ rect });
    const hash = stableHash({ rect, compositeInpaintMaskHash, compositeRasterLayerHash });
    const cachedGenerationMode = this.cache.generationModeCache.get(hash);

    if (cachedGenerationMode) {
      this.log.trace({ rect, cachedGenerationMode }, 'Using cached generation mode');
      return cachedGenerationMode;
    }

    const inpaintMaskImageData = canvasToImageData(this.getCompositeInpaintMaskCanvas(rect));
    const inpaintMaskTransparency = getImageDataTransparency(inpaintMaskImageData);
    const compositeLayerImageData = canvasToImageData(this.getCompositeRasterLayerCanvas(rect));
    const compositeLayerTransparency = getImageDataTransparency(compositeLayerImageData);

    let generationMode: GenerationMode;
    if (compositeLayerTransparency === 'FULLY_TRANSPARENT') {
      // When the initial image is fully transparent, we are always doing txt2img
      generationMode = 'txt2img';
    } else if (compositeLayerTransparency === 'PARTIALLY_TRANSPARENT') {
      // When the initial image is partially transparent, we are always outpainting
      generationMode = 'outpaint';
    } else if (inpaintMaskTransparency === 'FULLY_TRANSPARENT') {
      // compositeLayerTransparency === 'OPAQUE'
      // When the inpaint mask is fully transparent, we are doing img2img
      generationMode = 'img2img';
    } else {
      // Else at least some of the inpaint mask is opaque, so we are inpainting
      generationMode = 'inpaint';
    }

    this.cache.generationModeCache.set(hash, generationMode);
    return generationMode;
  }

  setCanvasManager = () => {
    this.log.debug('Setting canvas manager');
    $canvasManager.set(this);
  };

  getLoggingContext = (): SerializableObject => {
    return {
      path: this.path.join('.'),
    };
  };

  buildLogger = (getContext: () => SerializableObject): Logger => {
    return this.log.child((message) => {
      return {
        ...message,
        context: {
          ...message.context,
          ...getContext(),
        },
      };
    });
  };

  logDebugInfo() {
    // eslint-disable-next-line no-console
    console.log(this);
    for (const layer of this.rasterLayerAdapters.values()) {
      // eslint-disable-next-line no-console
      console.log(layer);
    }
  }

  getPrefixedId = getPrefixedId;
}
