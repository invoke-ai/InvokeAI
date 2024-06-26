import { getImageDataTransparency } from 'common/util/arrayBuffer';
import { CanvasBackground } from 'features/controlLayers/konva/renderers/background';
import {
  CanvasBbox,
  CanvasDocumentSizeOverlay,
  CanvasPreview,
  CanvasStagingArea,
  CanvasTool,
} from 'features/controlLayers/konva/renderers/preview';
import { konvaNodeToBlob, konvaNodeToImageData, previewBlob } from 'features/controlLayers/konva/util';
import type {
  BrushLineAddedArg,
  CanvasEntity,
  CanvasV2State,
  EraserLineAddedArg,
  GenerationMode,
  PointAddedToLineArg,
  PosChangedArg,
  Rect,
  RectShapeAddedArg,
  RgbaColor,
  StageAttrs,
  Tool,
} from 'features/controlLayers/store/types';
import { isValidLayer } from 'features/nodes/util/graph/generation/addLayers';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import { atom } from 'nanostores';
import { getImageDTO as defaultGetImageDTO, uploadImage as defaultUploadImage } from 'services/api/endpoints/images';
import type { ImageCategory, ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

import { CanvasControlAdapter } from './renderers/controlAdapters';
import { CanvasInpaintMask } from './renderers/inpaintMask';
import { CanvasLayer } from './renderers/layers';
import { CanvasRegion } from './renderers/regions';

export type StateApi = {
  getToolState: () => CanvasV2State['tool'];
  getCurrentFill: () => RgbaColor;
  setTool: (tool: Tool) => void;
  setToolBuffer: (tool: Tool | null) => void;
  getIsDrawing: () => boolean;
  setIsDrawing: (isDrawing: boolean) => void;
  getIsMouseDown: () => boolean;
  setIsMouseDown: (isMouseDown: boolean) => void;
  getLastMouseDownPos: () => Vector2d | null;
  setLastMouseDownPos: (pos: Vector2d | null) => void;
  getLastCursorPos: () => Vector2d | null;
  setLastCursorPos: (pos: Vector2d | null) => void;
  getLastAddedPoint: () => Vector2d | null;
  setLastAddedPoint: (pos: Vector2d | null) => void;
  setStageAttrs: (attrs: StageAttrs) => void;
  getSelectedEntity: () => CanvasEntity | null;
  getSpaceKey: () => boolean;
  setSpaceKey: (val: boolean) => void;
  getBbox: () => CanvasV2State['bbox'];
  getSettings: () => CanvasV2State['settings'];
  onBrushLineAdded: (arg: BrushLineAddedArg, entityType: CanvasEntity['type']) => void;
  onEraserLineAdded: (arg: EraserLineAddedArg, entityType: CanvasEntity['type']) => void;
  onPointAddedToLine: (arg: PointAddedToLineArg, entityType: CanvasEntity['type']) => void;
  onRectShapeAdded: (arg: RectShapeAddedArg, entityType: CanvasEntity['type']) => void;
  onBrushWidthChanged: (size: number) => void;
  onEraserWidthChanged: (size: number) => void;
  getMaskOpacity: () => number;
  onPosChanged: (arg: PosChangedArg, entityType: CanvasEntity['type']) => void;
  onBboxTransformed: (bbox: Rect) => void;
  getShiftKey: () => boolean;
  getCtrlKey: () => boolean;
  getMetaKey: () => boolean;
  getAltKey: () => boolean;
  getDocument: () => CanvasV2State['document'];
  getLayersState: () => CanvasV2State['layers'];
  getControlAdaptersState: () => CanvasV2State['controlAdapters'];
  getRegionsState: () => CanvasV2State['regions'];
  getInpaintMaskState: () => CanvasV2State['inpaintMask'];
  getStagingAreaState: () => CanvasV2State['stagingArea'];
  onInpaintMaskImageCached: (imageDTO: ImageDTO) => void;
  onRegionMaskImageCached: (id: string, imageDTO: ImageDTO) => void;
  onLayerImageCached: (imageDTO: ImageDTO) => void;
};

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

  constructor(
    stage: Konva.Stage,
    container: HTMLDivElement,
    stateApi: StateApi,
    getImageDTO: Util['getImageDTO'] = defaultGetImageDTO,
    uploadImage: Util['uploadImage'] = defaultUploadImage
  ) {
    this.stage = stage;
    this.container = container;
    this.stateApi = stateApi;
    this.util = {
      getImageDTO,
      uploadImage,
    };

    this.preview = new CanvasPreview(
      new CanvasBbox(
        this.stateApi.getBbox,
        this.stateApi.onBboxTransformed,
        this.stateApi.getShiftKey,
        this.stateApi.getCtrlKey,
        this.stateApi.getMetaKey,
        this.stateApi.getAltKey
      ),
      new CanvasTool(),
      new CanvasDocumentSizeOverlay(),
      new CanvasStagingArea()
    );
    this.stage.add(this.preview.konvaLayer);

    this.background = new CanvasBackground();
    this.stage.add(this.background.konvaLayer);

    this.inpaintMask = new CanvasInpaintMask(this.stateApi.getInpaintMaskState(), this.stateApi.onPosChanged);
    this.stage.add(this.inpaintMask.konvaLayer);

    this.layers = new Map();
    this.regions = new Map();
    this.controlAdapters = new Map();
  }

  renderLayers() {
    const { entities } = this.stateApi.getLayersState();
    const toolState = this.stateApi.getToolState();

    for (const adapter of this.layers.values()) {
      if (!entities.find((l) => l.id === adapter.id)) {
        adapter.destroy();
        this.layers.delete(adapter.id);
      }
    }

    for (const entity of entities) {
      let adapter = this.layers.get(entity.id);
      if (!adapter) {
        adapter = new CanvasLayer(entity, this.stateApi.onPosChanged);
        this.layers.set(adapter.id, adapter);
        this.stage.add(adapter.konvaLayer);
      }
      adapter.render(entity, toolState.selected);
    }
  }

  renderRegions() {
    const { entities } = this.stateApi.getRegionsState();
    const maskOpacity = this.stateApi.getMaskOpacity();
    const toolState = this.stateApi.getToolState();
    const selectedEntity = this.stateApi.getSelectedEntity();

    // Destroy the konva nodes for nonexistent entities
    for (const adapter of this.regions.values()) {
      if (!entities.find((rg) => rg.id === adapter.id)) {
        adapter.destroy();
        this.regions.delete(adapter.id);
      }
    }

    for (const entity of entities) {
      let adapter = this.regions.get(entity.id);
      if (!adapter) {
        adapter = new CanvasRegion(entity, this.stateApi.onPosChanged);
        this.regions.set(adapter.id, adapter);
        this.stage.add(adapter.konvaLayer);
      }
      adapter.render(entity, toolState.selected, selectedEntity, maskOpacity);
    }
  }

  renderInpaintMask() {
    const inpaintMaskState = this.stateApi.getInpaintMaskState();
    const toolState = this.stateApi.getToolState();
    const selectedEntity = this.stateApi.getSelectedEntity();
    const maskOpacity = this.stateApi.getMaskOpacity();

    this.inpaintMask.render(inpaintMaskState, toolState.selected, selectedEntity, maskOpacity);
  }

  renderControlAdapters() {
    const { entities } = this.stateApi.getControlAdaptersState();

    for (const adapter of this.controlAdapters.values()) {
      if (!entities.find((ca) => ca.id === adapter.id)) {
        adapter.destroy();
        this.controlAdapters.delete(adapter.id);
      }
    }

    for (const entity of entities) {
      let adapter = this.controlAdapters.get(entity.id);
      if (!adapter) {
        adapter = new CanvasControlAdapter(entity);
        this.controlAdapters.set(adapter.id, adapter);
        this.stage.add(adapter.konvaLayer);
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
    this.background.konvaLayer.zIndex(++zIndex);
    for (const layer of layers) {
      this.layers.get(layer.id)?.konvaLayer.zIndex(++zIndex);
    }
    for (const ca of controlAdapters) {
      this.controlAdapters.get(ca.id)?.konvaLayer.zIndex(++zIndex);
    }
    for (const rg of regions) {
      this.regions.get(rg.id)?.konvaLayer.zIndex(++zIndex);
    }
    this.inpaintMask?.konvaLayer.zIndex(++zIndex);
    this.preview.konvaLayer.zIndex(++zIndex);
  }

  renderDocumentSizeOverlay() {
    this.preview.documentSizeOverlay.render(this.stage, this.stateApi.getDocument());
  }

  renderBbox() {
    this.preview.bbox.render(this.stateApi.getBbox(), this.stateApi.getToolState());
  }

  renderToolPreview() {
    this.preview.tool.render(
      this.stage,
      1, // TODO(psyche): this should be renderable entity count
      this.stateApi.getToolState(),
      this.stateApi.getCurrentFill(),
      this.stateApi.getSelectedEntity(),
      this.stateApi.getLastCursorPos(),
      this.stateApi.getLastMouseDownPos(),
      this.stateApi.getIsDrawing(),
      this.stateApi.getIsMouseDown()
    );
  }

  renderBackground() {
    this.background.renderBackground(this.stage);
  }

  renderStagingArea() {
    this.preview.stagingArea.render(this.stateApi.getStagingAreaState());
  }

  fitDocument() {
    this.preview.documentSizeOverlay.fitToStage(this.stage, this.stateApi.getDocument(), this.stateApi.setStageAttrs);
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
    this.renderBackground();
    this.renderDocumentSizeOverlay();
  }

  getInpaintMaskLayerClone(): Konva.Layer {
    const layerClone = this.inpaintMask.konvaLayer.clone();
    const objectGroupClone = this.inpaintMask.konvaObjectGroup.clone();

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

    const layerClone = canvasRegion.konvaLayer.clone();
    const objectGroupClone = canvasRegion.konvaObjectGroup.clone();

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
