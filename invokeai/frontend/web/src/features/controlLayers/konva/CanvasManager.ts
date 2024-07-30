import type { Store } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { CanvasInitialImage } from 'features/controlLayers/konva/CanvasInitialImage';
import { CanvasProgressPreview } from 'features/controlLayers/konva/CanvasProgressPreview';
import {
  getCompositeLayerImage,
  getControlAdapterImage,
  getGenerationMode,
  getInitialImage,
  getInpaintMaskImage,
  getRegionMaskImage,
} from 'features/controlLayers/konva/util';
import type { Extents, ExtentsResult, GetBboxTask, WorkerLogMessage } from 'features/controlLayers/konva/worker';
import { $lastProgressEvent, $shouldShowStagedImage } from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasV2State, GenerationMode } from 'features/controlLayers/store/types';
import type Konva from 'konva';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { getImageDTO as defaultGetImageDTO, uploadImage as defaultUploadImage } from 'services/api/endpoints/images';
import type { ImageCategory, ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

import { CanvasBackground } from './CanvasBackground';
import { CanvasBbox } from './CanvasBbox';
import { CanvasControlAdapter } from './CanvasControlAdapter';
import { CanvasInpaintMask } from './CanvasInpaintMask';
import { CanvasLayer } from './CanvasLayer';
import { CanvasPreview } from './CanvasPreview';
import { CanvasRegion } from './CanvasRegion';
import { CanvasStagingArea } from './CanvasStagingArea';
import { CanvasStateApi } from './CanvasStateApi';
import { CanvasTool } from './CanvasTool';
import { setStageEventHandlers } from './events';

// type Extents = {
//   minX: number;
//   minY: number;
//   maxX: number;
//   maxY: number;
// };
// type GetBboxTask = {
//   id: string;
//   type: 'get_bbox';
//   data: { imageData: ImageData };
// };

// type GetBboxResult = {
//   id: string;
//   type: 'get_bbox';
//   data: { extents: Extents | null };
// };

type Util = {
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>;
  uploadImage: (
    blob: Blob,
    fileName: string,
    image_category: ImageCategory,
    is_intermediate: boolean
  ) => Promise<ImageDTO>;
};

export const $canvasManager = atom<CanvasManager | null>(null);

export class CanvasManager {
  private static BBOX_PADDING_PX = 5;
  static BBOX_DEBOUNCE_MS = 300;

  stage: Konva.Stage;
  container: HTMLDivElement;
  controlAdapters: Map<string, CanvasControlAdapter>;
  layers: Map<string, CanvasLayer>;
  regions: Map<string, CanvasRegion>;
  inpaintMask: CanvasInpaintMask;
  initialImage: CanvasInitialImage;
  util: Util;
  stateApi: CanvasStateApi;
  preview: CanvasPreview;
  background: CanvasBackground;

  log: Logger;
  workerLog: Logger;

  onTransform: ((isTransforming: boolean) => void) | null;

  private store: Store<RootState>;
  private isFirstRender: boolean;
  private prevState: CanvasV2State;
  private worker: Worker;
  private tasks: Map<string, { task: GetBboxTask; onComplete: (extents: Extents | null) => void }>;

  constructor(
    stage: Konva.Stage,
    container: HTMLDivElement,
    store: Store<RootState>,
    getImageDTO: Util['getImageDTO'] = defaultGetImageDTO,
    uploadImage: Util['uploadImage'] = defaultUploadImage
  ) {
    this.stage = stage;
    this.container = container;
    this.store = store;
    this.stateApi = new CanvasStateApi(this.store);
    this.prevState = this.stateApi.getState();
    this.isFirstRender = true;

    this.log = logger('canvas');
    this.workerLog = logger('worker');

    this.util = {
      getImageDTO,
      uploadImage,
    };

    this.preview = new CanvasPreview(
      new CanvasBbox(this),
      new CanvasTool(this),
      new CanvasStagingArea(this),
      new CanvasProgressPreview(this)
    );
    this.stage.add(this.preview.layer);

    this.background = new CanvasBackground(this);
    this.stage.add(this.background.konva.layer);

    this.inpaintMask = new CanvasInpaintMask(this.stateApi.getInpaintMaskState(), this);
    this.stage.add(this.inpaintMask.konva.layer);

    this.layers = new Map();
    this.regions = new Map();
    this.controlAdapters = new Map();

    this.initialImage = new CanvasInitialImage(this.stateApi.getInitialImageState(), this);
    this.stage.add(this.initialImage.konva.layer);

    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'worker' });
    this.tasks = new Map();
    this.worker.onmessage = (event: MessageEvent<ExtentsResult | WorkerLogMessage>) => {
      const { type, data } = event.data;
      if (type === 'log') {
        if (data.ctx) {
          this.workerLog[data.level](data.ctx, data.message);
        } else {
          this.workerLog[data.level](data.message);
        }
      } else if (type === 'extents') {
        const task = this.tasks.get(data.id);
        if (!task) {
          return;
        }
        task.onComplete(data.extents);
      }
    };
    this.worker.onerror = (event) => {
      this.log.error({ message: event.message }, 'Worker error');
    };
    this.worker.onmessageerror = () => {
      this.log.error('Worker message error');
    };
    this.onTransform = null;
  }

  getLogger(namespace: string) {
    const managerNamespace = this.log.getContext().namespace;
    return this.log.child({ namespace: `${managerNamespace}.${namespace}` });
  }

  requestBbox(data: Omit<GetBboxTask['data'], 'id'>, onComplete: (extents: Extents | null) => void) {
    const id = crypto.randomUUID();
    const task: GetBboxTask = {
      type: 'get_bbox',
      data: { ...data, id },
    };
    this.tasks.set(id, { task, onComplete });
    this.worker.postMessage(task, [data.buffer]);
  }

  async renderInitialImage() {
    await this.initialImage.render(this.stateApi.getInitialImageState());
  }

  async renderRegions() {
    const { entities } = this.stateApi.getRegionsState();

    // Destroy the konva nodes for nonexistent entities
    for (const canvasRegion of this.regions.values()) {
      if (!entities.find((rg) => rg.id === canvasRegion.id)) {
        canvasRegion.destroy();
        this.regions.delete(canvasRegion.id);
      }
    }

    for (const entity of entities) {
      let adapter = this.regions.get(entity.id);
      if (!adapter) {
        adapter = new CanvasRegion(entity, this);
        this.regions.set(adapter.id, adapter);
        this.stage.add(adapter.konva.layer);
      }
      await adapter.render(entity);
    }
  }

  async renderProgressPreview() {
    await this.preview.progressPreview.render(this.stateApi.getLastProgressEvent());
  }

  async renderInpaintMask() {
    const inpaintMaskState = this.stateApi.getInpaintMaskState();
    await this.inpaintMask.render(inpaintMaskState);
  }

  async renderControlAdapters() {
    const { entities } = this.stateApi.getControlAdaptersState();

    for (const canvasControlAdapter of this.controlAdapters.values()) {
      if (!entities.find((ca) => ca.id === canvasControlAdapter.id)) {
        canvasControlAdapter.destroy();
        this.controlAdapters.delete(canvasControlAdapter.id);
      }
    }

    for (const entity of entities) {
      let adapter = this.controlAdapters.get(entity.id);
      if (!adapter) {
        adapter = new CanvasControlAdapter(entity, this);
        this.controlAdapters.set(adapter.id, adapter);
        this.stage.add(adapter.konva.layer);
      }
      await adapter.render(entity);
    }
  }

  syncStageScale() {
    for (const layer of this.layers.values()) {
      layer.syncStageScale();
    }
  }

  arrangeEntities() {
    const { getLayersState, getControlAdaptersState, getRegionsState } = this.stateApi;
    const layers = getLayersState().entities;
    const controlAdapters = getControlAdaptersState().entities;
    const regions = getRegionsState().entities;
    let zIndex = 0;
    this.background.konva.layer.zIndex(++zIndex);
    this.initialImage.konva.layer.zIndex(++zIndex);
    for (const layer of layers) {
      this.layers.get(layer.id)?.konva.layer.zIndex(++zIndex);
    }
    for (const ca of controlAdapters) {
      this.controlAdapters.get(ca.id)?.konva.layer.zIndex(++zIndex);
    }
    for (const rg of regions) {
      this.regions.get(rg.id)?.konva.layer.zIndex(++zIndex);
    }
    this.inpaintMask.konva.layer.zIndex(++zIndex);
    this.preview.layer.zIndex(++zIndex);
  }

  fitStageToContainer() {
    this.stage.width(this.container.offsetWidth);
    this.stage.height(this.container.offsetHeight);
    this.stateApi.setStageAttrs({
      position: { x: this.stage.x(), y: this.stage.y() },
      dimensions: { width: this.stage.width(), height: this.stage.height() },
      scale: this.stage.scaleX(),
    });
    this.background.render();
  }

  getTransformingLayer() {
    return Array.from(this.layers.values()).find((layer) => layer.isTransforming);
  }

  getIsTransforming() {
    return Boolean(this.getTransformingLayer());
  }

  startTransform() {
    if (this.getIsTransforming()) {
      return;
    }
    const layer = this.getSelectedEntityAdapter();
    assert(layer instanceof CanvasLayer, 'No selected layer');
    layer.startTransform();
    this.onTransform?.(true);
  }

  applyTransform() {
    const layer = this.getTransformingLayer();
    if (layer) {
      layer.applyTransform();
    }
    this.onTransform?.(false);
  }

  cancelTransform() {
    const layer = this.getTransformingLayer();
    if (layer) {
      layer.cancelTransform();
    }
    this.onTransform?.(false);
  }

  render = async () => {
    const state = this.stateApi.getState();

    if (this.prevState === state && !this.isFirstRender) {
      this.log.trace('No changes detected, skipping render');
      return;
    }

    if (this.isFirstRender || state.layers.entities !== this.prevState.layers.entities) {
      this.log.debug('Rendering layers');

      for (const canvasLayer of this.layers.values()) {
        if (!state.layers.entities.find((l) => l.id === canvasLayer.id)) {
          this.log.debug(`Destroying deleted layer ${canvasLayer.id}`);
          canvasLayer.destroy();
          this.layers.delete(canvasLayer.id);
        }
      }

      for (const entityState of state.layers.entities) {
        let adapter = this.layers.get(entityState.id);
        if (!adapter) {
          this.log.debug(`Creating layer layer ${entityState.id}`);
          adapter = new CanvasLayer(entityState, this);
          this.layers.set(adapter.id, adapter);
          this.stage.add(adapter.konva.layer);
        }
        await adapter.update({
          state: entityState,
          toolState: state.tool,
          isSelected: state.selectedEntityIdentifier?.id === entityState.id,
        });
      }
    }

    if (
      this.isFirstRender ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log.debug('Updating interaction');
      for (const layer of this.layers.values()) {
        layer.updateInteraction({ toolState: state.tool, isSelected: state.selectedEntityIdentifier?.id === layer.id });
      }
    }

    if (
      this.isFirstRender ||
      state.initialImage !== this.prevState.initialImage ||
      state.bbox.rect !== this.prevState.bbox.rect ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log.debug('Rendering initial image');
      await this.renderInitialImage();
    }

    if (
      this.isFirstRender ||
      state.regions.entities !== this.prevState.regions.entities ||
      state.settings.maskOpacity !== this.prevState.settings.maskOpacity ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log.debug('Rendering regions');
      await this.renderRegions();
    }

    if (
      this.isFirstRender ||
      state.inpaintMask !== this.prevState.inpaintMask ||
      state.settings.maskOpacity !== this.prevState.settings.maskOpacity ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log.debug('Rendering inpaint mask');
      await this.renderInpaintMask();
    }

    if (
      this.isFirstRender ||
      state.controlAdapters.entities !== this.prevState.controlAdapters.entities ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log.debug('Rendering control adapters');
      await this.renderControlAdapters();
    }

    if (
      this.isFirstRender ||
      state.bbox !== this.prevState.bbox ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.session.isActive !== this.prevState.session.isActive
    ) {
      this.log.debug('Rendering generation bbox');
      await this.preview.bbox.render();
    }

    if (
      this.isFirstRender ||
      state.layers !== this.prevState.layers ||
      state.controlAdapters !== this.prevState.controlAdapters ||
      state.regions !== this.prevState.regions
    ) {
      // this.log.debug('Updating entity bboxes');
      // debouncedUpdateBboxes(stage, canvasV2.layers, canvasV2.controlAdapters, canvasV2.regions, onBboxChanged);
    }

    if (this.isFirstRender || state.session !== this.prevState.session) {
      this.log.debug('Rendering staging area');
      await this.preview.stagingArea.render();
    }

    if (
      this.isFirstRender ||
      state.layers.entities !== this.prevState.layers.entities ||
      state.controlAdapters.entities !== this.prevState.controlAdapters.entities ||
      state.regions.entities !== this.prevState.regions.entities ||
      state.inpaintMask !== this.prevState.inpaintMask ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log.debug('Arranging entities');
      await this.arrangeEntities();
    }

    this.prevState = state;

    if (this.isFirstRender) {
      this.isFirstRender = false;
    }
  };

  initialize = () => {
    this.log.debug('Initializing renderer');
    this.stage.container(this.container);

    const cleanupListeners = setStageEventHandlers(this);

    // We can use a resize observer to ensure the stage always fits the container. We also need to re-render the bg and
    // document bounds overlay when the stage is resized.
    const resizeObserver = new ResizeObserver(this.fitStageToContainer.bind(this));
    resizeObserver.observe(this.container);
    this.fitStageToContainer();

    const unsubscribeRenderer = this.store.subscribe(this.render);

    // When we this flag, we need to render the staging area
    $shouldShowStagedImage.subscribe(async (shouldShowStagedImage, prevShouldShowStagedImage) => {
      if (shouldShowStagedImage !== prevShouldShowStagedImage) {
        this.log.debug('Rendering staging area');
        await this.preview.stagingArea.render();
      }
    });

    $lastProgressEvent.subscribe(async (lastProgressEvent, prevLastProgressEvent) => {
      if (lastProgressEvent !== prevLastProgressEvent) {
        this.log.debug('Rendering progress image');
        await this.preview.progressPreview.render(lastProgressEvent);
      }
    });

    this.log.debug('First render of konva stage');
    this.preview.tool.render();
    this.render();

    return () => {
      this.log.debug('Cleaning up konva renderer');
      unsubscribeRenderer();
      cleanupListeners();
      $shouldShowStagedImage.off();
      resizeObserver.disconnect();
    };
  };

  getStageScale(): number {
    // The stage is never scaled differently in x and y
    return this.stage.scaleX();
  }

  getScaledPixel(): number {
    return 1 / this.getStageScale();
  }

  getScaledBboxPadding(): number {
    return CanvasManager.BBOX_PADDING_PX / this.getStageScale();
  }

  getTransformerPadding(): number {
    return CanvasManager.BBOX_PADDING_PX;
  }

  getSelectedEntityAdapter = (): CanvasLayer | CanvasRegion | CanvasControlAdapter | CanvasInpaintMask | null => {
    const state = this.stateApi.getState();
    const identifier = state.selectedEntityIdentifier;
    if (!identifier) {
      return null;
    } else if (identifier.type === 'layer') {
      return this.layers.get(identifier.id) ?? null;
    } else if (identifier.type === 'control_adapter') {
      return this.controlAdapters.get(identifier.id) ?? null;
    } else if (identifier.type === 'regional_guidance') {
      return this.regions.get(identifier.id) ?? null;
    } else if (identifier.type === 'inpaint_mask') {
      return this.inpaintMask;
    } else {
      return null;
    }
  };

  getGenerationMode(): GenerationMode {
    const session = this.stateApi.getSession();
    if (session.isActive) {
      return getGenerationMode({ manager: this });
    }

    const initialImageState = this.stateApi.getInitialImageState();

    if (initialImageState.imageObject && initialImageState.isEnabled) {
      return 'img2img';
    }

    return 'txt2img';
  }

  getControlAdapterImage(arg: Omit<Parameters<typeof getControlAdapterImage>[0], 'manager'>) {
    return getControlAdapterImage({ ...arg, manager: this });
  }

  getRegionMaskImage(arg: Omit<Parameters<typeof getRegionMaskImage>[0], 'manager'>) {
    return getRegionMaskImage({ ...arg, manager: this });
  }

  getInpaintMaskImage(arg: Omit<Parameters<typeof getInpaintMaskImage>[0], 'manager'>) {
    return getInpaintMaskImage({ ...arg, manager: this });
  }

  getInitialImage(arg: Omit<Parameters<typeof getCompositeLayerImage>[0], 'manager'>) {
    if (this.stateApi.getSession().isActive) {
      return getCompositeLayerImage({ ...arg, manager: this });
    } else {
      return getInitialImage({ ...arg, manager: this });
    }
  }

  logDebugInfo() {
    console.log(this);
    for (const layer of this.layers.values()) {
      console.log(layer);
    }
  }
}
