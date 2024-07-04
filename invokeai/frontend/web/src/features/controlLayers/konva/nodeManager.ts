import type { Store } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { getImageDataTransparency } from 'common/util/arrayBuffer';
import { CanvasBackground } from 'features/controlLayers/konva/background';
import { setStageEventHandlers } from 'features/controlLayers/konva/events';
import { CanvasPreview } from 'features/controlLayers/konva/preview';
import { konvaNodeToBlob, konvaNodeToImageData, previewBlob } from 'features/controlLayers/konva/util';
import { $lastProgressEvent, $shouldShowStagedImage } from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasV2State, GenerationMode, Rect } from 'features/controlLayers/store/types';
import { isValidLayer } from 'features/nodes/util/graph/generation/addLayers';
import type Konva from 'konva';
import { atom } from 'nanostores';
import { getImageDTO as defaultGetImageDTO, uploadImage as defaultUploadImage } from 'services/api/endpoints/images';
import type { ImageCategory, ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

import { CanvasBbox } from './bbox';
import { CanvasControlAdapter } from './controlAdapters';
import { CanvasDocumentSizeOverlay } from './documentSizeOverlay';
import { CanvasInpaintMask } from './inpaintMask';
import { CanvasLayer } from './layers';
import { CanvasRegion } from './regions';
import { CanvasStagingArea } from './stagingArea';
import { StateApi } from './stateApi';
import { CanvasTool } from './tool';

type Util = {
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>;
  uploadImage: (
    blob: Blob,
    fileName: string,
    image_category: ImageCategory,
    is_intermediate: boolean
  ) => Promise<ImageDTO>;
};

const $nodeManager = atom<KonvaNodeManager | null>(null);
export function getNodeManager() {
  const nodeManager = $nodeManager.get();
  assert(nodeManager !== null, 'Node manager not initialized');
  return nodeManager;
}
export function setNodeManager(nodeManager: KonvaNodeManager) {
  $nodeManager.set(nodeManager);
}

export class KonvaNodeManager {
  stage: Konva.Stage;
  container: HTMLDivElement;
  controlAdapters: Map<string, CanvasControlAdapter>;
  layers: Map<string, CanvasLayer>;
  regions: Map<string, CanvasRegion>;
  inpaintMask: CanvasInpaintMask;
  util: Util;
  stateApi: StateApi;
  preview: CanvasPreview;
  background: CanvasBackground;
  private store: Store<RootState>;
  private isFirstRender: boolean;
  private prevState: CanvasV2State;
  private log: (message: string) => void;

  constructor(
    stage: Konva.Stage,
    container: HTMLDivElement,
    store: Store<RootState>,
    log: (message: string) => void,
    getImageDTO: Util['getImageDTO'] = defaultGetImageDTO,
    uploadImage: Util['uploadImage'] = defaultUploadImage
  ) {
    this.log = log;
    this.stage = stage;
    this.container = container;
    this.store = store;
    this.stateApi = new StateApi(this.store, this.log);
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

    this.inpaintMask = new CanvasInpaintMask(this);
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
    this.background.renderBackground();
    this.preview.documentSizeOverlay.render();
  }

  render = async () => {
    const state = this.stateApi.getState();

    if (this.prevState === state && !this.isFirstRender) {
      this.log('No changes detected, skipping render');
      return;
    }

    if (
      this.isFirstRender ||
      state.layers.entities !== this.prevState.layers.entities ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log('Rendering layers');
      this.renderLayers();
    }

    if (
      this.isFirstRender ||
      state.regions.entities !== this.prevState.regions.entities ||
      state.settings.maskOpacity !== this.prevState.settings.maskOpacity ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log('Rendering regions');
      this.renderRegions();
    }

    if (
      this.isFirstRender ||
      state.inpaintMask !== this.prevState.inpaintMask ||
      state.settings.maskOpacity !== this.prevState.settings.maskOpacity ||
      state.tool.selected !== this.prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log('Rendering inpaint mask');
      this.renderInpaintMask();
    }

    if (
      this.isFirstRender ||
      state.controlAdapters.entities !== this.prevState.controlAdapters.entities ||
      state.selectedEntityIdentifier?.id !== this.prevState.selectedEntityIdentifier?.id
    ) {
      this.log('Rendering control adapters');
      this.renderControlAdapters();
    }

    if (this.isFirstRender || state.document !== this.prevState.document) {
      this.log('Rendering document bounds overlay');
      this.preview.documentSizeOverlay.render();
    }

    if (
      this.isFirstRender ||
      state.bbox !== this.prevState.bbox ||
      state.tool.selected !== this.prevState.tool.selected
    ) {
      this.log('Rendering generation bbox');
      this.preview.bbox.render();
    }

    if (
      this.isFirstRender ||
      state.layers !== this.prevState.layers ||
      state.controlAdapters !== this.prevState.controlAdapters ||
      state.regions !== this.prevState.regions
    ) {
      // this.log('Updating entity bboxes');
      // debouncedUpdateBboxes(stage, canvasV2.layers, canvasV2.controlAdapters, canvasV2.regions, onBboxChanged);
    }

    if (this.isFirstRender || state.stagingArea !== this.prevState.stagingArea) {
      this.log('Rendering staging area');
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
      this.log('Arranging entities');
      this.arrangeEntities();
    }

    this.prevState = state;

    if (this.isFirstRender) {
      this.isFirstRender = false;
    }
  };

  initialize = () => {
    this.log('Initializing renderer');
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
      this.log('Rendering staging area');
      if (shouldShowStagedImage !== prevShouldShowStagedImage) {
        this.preview.stagingArea.render();
      }
    });

    $lastProgressEvent.subscribe(() => {
      this.log('Rendering staging area');
      this.preview.stagingArea.render();
    });

    this.log('First render of konva stage');
    // On first render, the document should be fit to the stage.
    this.preview.documentSizeOverlay.render();
    this.preview.documentSizeOverlay.fitToStage();
    this.preview.tool.render();
    this.render();

    return () => {
      this.log('Cleaning up konva renderer');
      unsubscribeRenderer();
      cleanupListeners();
      $shouldShowStagedImage.off();
      resizeObserver.disconnect();
    };
  };

  getInpaintMaskLayerClone(): Konva.Layer {
    const layerClone = this.inpaintMask.layer.clone();
    const objectGroupClone = this.inpaintMask.group.clone();

    layerClone.destroyChildren();
    layerClone.add(objectGroupClone);

    objectGroupClone.opacity(1);
    objectGroupClone.cache();

    return layerClone;
  }

  getRegionMaskLayerClone(arg: { id: string }): Konva.Layer {
    const { id } = arg;

    const canvasRegion = this.regions.get(id);
    assert(canvasRegion, `Canvas region with id ${id} not found`);

    const layerClone = canvasRegion.layer.clone();
    const objectGroupClone = canvasRegion.group.clone();

    layerClone.destroyChildren();
    layerClone.add(objectGroupClone);

    objectGroupClone.opacity(1);
    objectGroupClone.cache();

    return layerClone;
  }

  getCompositeLayerStageClone(): Konva.Stage {
    const layersState = this.stateApi.getLayersState();

    const stageClone = this.stage.clone();

    stageClone.scaleX(1);
    stageClone.scaleY(1);
    stageClone.x(0);
    stageClone.y(0);

    const validLayers = layersState.entities.filter(isValidLayer);

    // Konva bug (?) - when iterating over the array returned from `stage.getLayers()`, if you destroy a layer, the array
    // is mutated in-place and the next iteration will skip the next layer. To avoid this, we first collect the layers
    // to delete in a separate array and then destroy them.
    // TODO(psyche): Maybe report this?
    const toDelete: Konva.Layer[] = [];

    for (const konvaLayer of stageClone.getLayers()) {
      const layer = validLayers.find((l) => l.id === konvaLayer.id());
      if (!layer) {
        toDelete.push(konvaLayer);
      }
    }

    for (const konvaLayer of toDelete) {
      konvaLayer.destroy();
    }

    return stageClone;
  }

  getGenerationMode(): GenerationMode {
    const { x, y, width, height } = this.stateApi.getBbox();
    const inpaintMaskLayer = this.getInpaintMaskLayerClone();
    const inpaintMaskImageData = konvaNodeToImageData(inpaintMaskLayer, { x, y, width, height });
    const inpaintMaskTransparency = getImageDataTransparency(inpaintMaskImageData);
    const compositeLayer = this.getCompositeLayerStageClone();
    const compositeLayerImageData = konvaNodeToImageData(compositeLayer, { x, y, width, height });
    const compositeLayerTransparency = getImageDataTransparency(compositeLayerImageData);
    if (compositeLayerTransparency.isPartiallyTransparent) {
      if (compositeLayerTransparency.isFullyTransparent) {
        return 'txt2img';
      }
      return 'outpaint';
    } else {
      if (!inpaintMaskTransparency.isFullyTransparent) {
        return 'inpaint';
      }
      return 'img2img';
    }
  }

  async getRegionMaskImage(arg: { id: string; bbox?: Rect; preview?: boolean }): Promise<ImageDTO> {
    const { id, bbox, preview = false } = arg;
    const region = this.stateApi.getRegionsState().entities.find((entity) => entity.id === id);
    assert(region, `Region entity state with id ${id} not found`);

    // if (region.imageCache) {
    //   const imageDTO = await this.util.getImageDTO(region.imageCache.name);
    //   if (imageDTO) {
    //     return imageDTO;
    //   }
    // }

    const layerClone = this.getRegionMaskLayerClone({ id });
    const blob = await konvaNodeToBlob(layerClone, bbox);

    if (preview) {
      previewBlob(blob, `region ${region.id} mask`);
    }

    layerClone.destroy();

    const imageDTO = await this.util.uploadImage(blob, `${region.id}_mask.png`, 'mask', true);
    this.stateApi.onRegionMaskImageCached(region.id, imageDTO);
    return imageDTO;
  }

  async getInpaintMaskImage(arg: { bbox?: Rect; preview?: boolean }): Promise<ImageDTO> {
    const { bbox, preview = false } = arg;
    // const inpaintMask = this.stateApi.getInpaintMaskState();

    // if (inpaintMask.imageCache) {
    //   const imageDTO = await this.util.getImageDTO(inpaintMask.imageCache.name);
    //   if (imageDTO) {
    //     return imageDTO;
    //   }
    // }

    const layerClone = this.getInpaintMaskLayerClone();
    const blob = await konvaNodeToBlob(layerClone, bbox);

    if (preview) {
      previewBlob(blob, 'inpaint mask');
    }

    layerClone.destroy();

    const imageDTO = await this.util.uploadImage(blob, 'inpaint_mask.png', 'mask', true);
    this.stateApi.onInpaintMaskImageCached(imageDTO);
    return imageDTO;
  }

  async getImageSourceImage(arg: { bbox?: Rect; preview?: boolean }): Promise<ImageDTO> {
    const { bbox, preview = false } = arg;
    // const { imageCache } = this.stateApi.getLayersState();

    // if (imageCache) {
    //   const imageDTO = await this.util.getImageDTO(imageCache.name);
    //   if (imageDTO) {
    //     return imageDTO;
    //   }
    // }

    const stageClone = this.getCompositeLayerStageClone();

    const blob = await konvaNodeToBlob(stageClone, bbox);

    if (preview) {
      previewBlob(blob, 'image source');
    }

    stageClone.destroy();

    const imageDTO = await this.util.uploadImage(blob, 'base_layer.png', 'general', true);
    this.stateApi.onLayerImageCached(imageDTO);
    return imageDTO;
  }
}
