import type { Store } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import {
  getGenerationMode,
  getImageSourceImage,
  getInpaintMaskImage,
  getRegionMaskImage,
} from 'features/controlLayers/konva/util';
import { $lastProgressEvent, $shouldShowStagedImage } from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import type Konva from 'konva';
import { atom } from 'nanostores';
import { getImageDTO as defaultGetImageDTO, uploadImage as defaultUploadImage } from 'services/api/endpoints/images';
import type { ImageCategory, ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

import { CanvasBackground } from './CanvasBackground';
import { CanvasBbox } from './CanvasBbox';
import { CanvasControlAdapter } from './CanvasControlAdapter';
import { CanvasDocumentSizeOverlay } from './CanvasDocumentSizeOverlay';
import { CanvasInpaintMask } from './CanvasInpaintMask';
import { CanvasLayer } from './CanvasLayer';
import { CanvasPreview } from './CanvasPreview';
import { CanvasRegion } from './CanvasRegion';
import { CanvasStagingArea } from './CanvasStagingArea';
import { CanvasStateApi } from './CanvasStateApi';
import { CanvasTool } from './CanvasTool';
import { setStageEventHandlers } from './events';

const log = logger('canvas');

type Util = {
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>;
  uploadImage: (
    blob: Blob,
    fileName: string,
    image_category: ImageCategory,
    is_intermediate: boolean
  ) => Promise<ImageDTO>;
};

const $canvasManager = atom<CanvasManager | null>(null);
export function getCanvasManager() {
  const nodeManager = $canvasManager.get();
  assert(nodeManager !== null, 'Node manager not initialized');
  return nodeManager;
}
export function setCanvasManager(nodeManager: CanvasManager) {
  $canvasManager.set(nodeManager);
}

export class CanvasManager {
  stage: Konva.Stage;
  container: HTMLDivElement;
  controlAdapters: Map<string, CanvasControlAdapter>;
  layers: Map<string, CanvasLayer>;
  regions: Map<string, CanvasRegion>;
  inpaintMask: CanvasInpaintMask;
  util: Util;
  stateApi: CanvasStateApi;
  preview: CanvasPreview;
  background: CanvasBackground;
  private store: Store<RootState>;
  private isFirstRender: boolean;
  private prevState: CanvasV2State;

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

    this.util = {
      getImageDTO,
      uploadImage,
    };

    this.preview = new CanvasPreview(
      new CanvasBbox(this),
      new CanvasTool(this),
      new CanvasDocumentSizeOverlay(this),
      new CanvasStagingArea(this)
    );
    this.stage.add(this.preview.layer);

    this.background = new CanvasBackground(this);
    this.stage.add(this.background.layer);

    this.inpaintMask = new CanvasInpaintMask(this.stateApi.getInpaintMaskState(), this);
    this.stage.add(this.inpaintMask.layer);

    this.layers = new Map();
    this.regions = new Map();
    this.controlAdapters = new Map();
  }

  async renderLayers() {
    const { entities } = this.stateApi.getLayersState();

    for (const canvasLayer of this.layers.values()) {
      if (!entities.find((l) => l.id === canvasLayer.id)) {
        canvasLayer.destroy();
        this.layers.delete(canvasLayer.id);
      }
    }

    for (const entity of entities) {
      let adapter = this.layers.get(entity.id);
      if (!adapter) {
        adapter = new CanvasLayer(entity, this);
        this.layers.set(adapter.id, adapter);
        this.stage.add(adapter.layer);
      }
      await adapter.render(entity);
    }
  }

  renderRegions() {
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
        this.stage.add(adapter.layer);
      }
      adapter.render(entity);
    }
  }

  renderInpaintMask() {
    const inpaintMaskState = this.stateApi.getInpaintMaskState();
    this.inpaintMask.render(inpaintMaskState);
  }

  renderControlAdapters() {
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
        adapter = new CanvasControlAdapter(entity);
        this.controlAdapters.set(adapter.id, adapter);
        this.stage.add(adapter.layer);
      }
      adapter.render(entity);
    }
  }

  arrangeEntities() {
    const { getLayersState, getControlAdaptersState, getRegionsState } = this.stateApi;
    const layers = getLayersState().entities;
    const controlAdapters = getControlAdaptersState().entities;
    const regions = getRegionsState().entities;
    let zIndex = 0;
    this.background.layer.zIndex(++zIndex);
    for (const layer of layers) {
      this.layers.get(layer.id)?.layer.zIndex(++zIndex);
    }
    for (const ca of controlAdapters) {
      this.controlAdapters.get(ca.id)?.layer.zIndex(++zIndex);
    }
    for (const rg of regions) {
      this.regions.get(rg.id)?.layer.zIndex(++zIndex);
    }
    this.inpaintMask.layer.zIndex(++zIndex);
    this.preview.layer.zIndex(++zIndex);
  }

  fitStageToContainer() {
    this.stage.width(this.container.offsetWidth);
    this.stage.height(this.container.offsetHeight);
    this.stateApi.setStageAttrs({
      x: this.stage.x(),
      y: this.stage.y(),
      width: this.stage.width(),
      height: this.stage.height(),
      scale: this.stage.scaleX(),
    });
    this.background.render();
    this.preview.documentSizeOverlay.render();
  }

  render = async () => {
    const state = this.stateApi.getState();

    if (this.prevState === state && !this.isFirstRender) {
      log.debug('No changes detected, skipping render');
      return;
    }

    if (
      this.isFirstRender ||
      state.layers.entities !== this.prevState.layers.entities ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      log.debug('Rendering layers');
      this.renderLayers();
    }

    if (
      this.isFirstRender ||
      state.regions.entities !== this.prevState.regions.entities ||
      state.settings.maskOpacity !== this.prevState.settings.maskOpacity ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      log.debug('Rendering regions');
      this.renderRegions();
    }

    if (
      this.isFirstRender ||
      state.inpaintMask !== this.prevState.inpaintMask ||
      state.settings.maskOpacity !== this.prevState.settings.maskOpacity ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      log.debug('Rendering inpaint mask');
      this.renderInpaintMask();
    }

    if (
      this.isFirstRender ||
      state.controlAdapters.entities !== this.prevState.controlAdapters.entities ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      log.debug('Rendering control adapters');
      this.renderControlAdapters();
    }

    if (this.isFirstRender || state.document !== this.prevState.document) {
      log.debug('Rendering document bounds overlay');
      this.preview.documentSizeOverlay.render();
    }

    if (
      this.isFirstRender ||
      state.bbox !== this.prevState.bbox ||
      state.tool.selected !== this.prevState.tool.selected
    ) {
      log.debug('Rendering generation bbox');
      this.preview.bbox.render();
    }

    if (
      this.isFirstRender ||
      state.layers !== this.prevState.layers ||
      state.controlAdapters !== this.prevState.controlAdapters ||
      state.regions !== this.prevState.regions
    ) {
      // log.debug('Updating entity bboxes');
      // debouncedUpdateBboxes(stage, canvasV2.layers, canvasV2.controlAdapters, canvasV2.regions, onBboxChanged);
    }

    if (this.isFirstRender || state.stagingArea !== this.prevState.stagingArea) {
      log.debug('Rendering staging area');
      this.preview.stagingArea.render();
    }

    if (
      this.isFirstRender ||
      state.layers.entities !== this.prevState.layers.entities ||
      state.controlAdapters.entities !== this.prevState.controlAdapters.entities ||
      state.regions.entities !== this.prevState.regions.entities ||
      state.inpaintMask !== this.prevState.inpaintMask ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      log.debug('Arranging entities');
      this.arrangeEntities();
    }

    this.prevState = state;

    if (this.isFirstRender) {
      this.isFirstRender = false;
    }
  };

  initialize = () => {
    log.debug('Initializing renderer');
    this.stage.container(this.container);

    const cleanupListeners = setStageEventHandlers(this);

    // We can use a resize observer to ensure the stage always fits the container. We also need to re-render the bg and
    // document bounds overlay when the stage is resized.
    const resizeObserver = new ResizeObserver(this.fitStageToContainer.bind(this));
    resizeObserver.observe(this.container);
    this.fitStageToContainer();

    const unsubscribeRenderer = this.store.subscribe(this.render);

    // When we this flag, we need to render the staging area
    $shouldShowStagedImage.subscribe((shouldShowStagedImage, prevShouldShowStagedImage) => {
      log.debug('Rendering staging area');
      if (shouldShowStagedImage !== prevShouldShowStagedImage) {
        this.preview.stagingArea.render();
      }
    });

    $lastProgressEvent.subscribe(() => {
      log.debug('Rendering staging area');
      this.preview.stagingArea.render();
    });

    log.debug('First render of konva stage');
    // On first render, the document should be fit to the stage.
    this.preview.documentSizeOverlay.render();
    this.preview.documentSizeOverlay.fitToStage();
    this.preview.tool.render();
    this.render();

    return () => {
      log.debug('Cleaning up konva renderer');
      unsubscribeRenderer();
      cleanupListeners();
      $shouldShowStagedImage.off();
      resizeObserver.disconnect();
    };
  };

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

  getGenerationMode() {
    return getGenerationMode({ manager: this });
  }

  getRegionMaskImage(arg: Omit<Parameters<typeof getRegionMaskImage>[0], 'manager'>) {
    return getRegionMaskImage({ ...arg, manager: this });
  }

  getInpaintMaskImage(arg: Omit<Parameters<typeof getInpaintMaskImage>[0], 'manager'>) {
    return getInpaintMaskImage({ ...arg, manager: this });
  }

  getImageSourceImage(arg: Omit<Parameters<typeof getImageSourceImage>[0], 'manager'>) {
    return getImageSourceImage({ ...arg, manager: this });
  }
}
