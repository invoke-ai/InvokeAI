import { getImageDataTransparency } from 'common/util/arrayBuffer';
import { konvaNodeToBlob, konvaNodeToImageData, previewBlob } from 'features/controlLayers/konva/util';
import type {
  BrushLine,
  BrushLineAddedArg,
  CanvasEntity,
  CanvasV2State,
  EraserLine,
  EraserLineAddedArg,
  GenerationMode,
  ImageObject,
  PointAddedToLineArg,
  PosChangedArg,
  Rect,
  RectShape,
  RectShapeAddedArg,
  RgbaColor,
  StageAttrs,
  Tool,
} from 'features/controlLayers/store/types';
import { isValidLayer } from 'features/nodes/util/graph/generation/addLayers';
import type Konva from 'konva';
import type { Vector2d } from 'konva/lib/types';
import { getImageDTO as defaultGetImageDTO, uploadImage as defaultUploadImage } from 'services/api/endpoints/images';
import type { ImageCategory, ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

export type BrushLineObjectRecord = {
  id: string;
  type: BrushLine['type'];
  konvaLine: Konva.Line;
  konvaLineGroup: Konva.Group;
};

export type EraserLineObjectRecord = {
  id: string;
  type: EraserLine['type'];
  konvaLine: Konva.Line;
  konvaLineGroup: Konva.Group;
};

export type RectShapeObjectRecord = {
  id: string;
  type: RectShape['type'];
  konvaRect: Konva.Rect;
};

export type ImageObjectRecord = {
  id: string;
  type: ImageObject['type'];
  konvaImageGroup: Konva.Group;
  konvaPlaceholderGroup: Konva.Group;
  konvaPlaceholderRect: Konva.Rect;
  konvaPlaceholderText: Konva.Text;
  konvaImage: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  isLoading: boolean;
  isError: boolean;
};

type ObjectRecord = BrushLineObjectRecord | EraserLineObjectRecord | RectShapeObjectRecord | ImageObjectRecord;

type KonvaApi = {
  renderRegions: () => void;
  renderLayers: () => void;
  renderControlAdapters: () => void;
  renderInpaintMask: () => void;
  renderBbox: () => void;
  renderDocumentOverlay: () => void;
  renderBackground: () => void;
  renderToolPreview: () => void;
  arrangeEntities: () => void;
  fitDocumentToStage: () => void;
  fitStageToContainer: () => void;
};

type BackgroundLayer = {
  layer: Konva.Layer;
};

type PreviewLayer = {
  layer: Konva.Layer;
  bbox: {
    group: Konva.Group;
    rect: Konva.Rect;
    transformer: Konva.Transformer;
  };
  tool: {
    group: Konva.Group;
    brush: {
      group: Konva.Group;
      fill: Konva.Circle;
      innerBorder: Konva.Circle;
      outerBorder: Konva.Circle;
    };
    rect: {
      rect: Konva.Rect;
    };
  };
  documentOverlay: {
    group: Konva.Group;
    innerRect: Konva.Rect;
    outerRect: Konva.Rect;
  };
};

type StateApi = {
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
  getRegionMaskImage: (arg: { id: string; bbox?: Rect; preview?: boolean }) => Promise<ImageDTO>;
  getInpaintMaskImage: (arg: { bbox?: Rect; preview?: boolean }) => Promise<ImageDTO>;
  getImageSourceImage: (arg: { bbox?: Rect; preview?: boolean }) => Promise<ImageDTO>;
  getMaskLayerClone: (arg: { id: string }) => Konva.Layer;
  getCompositeLayerStageClone: () => Konva.Stage;
  getGenerationMode: () => GenerationMode;
};

export class KonvaNodeManager {
  stage: Konva.Stage;
  container: HTMLDivElement;
  adapters: Map<string, KonvaEntityAdapter>;
  util: Util;
  _background: BackgroundLayer | null;
  _preview: PreviewLayer | null;
  _konvaApi: KonvaApi | null;
  _stateApi: StateApi | null;

  constructor(
    stage: Konva.Stage,
    container: HTMLDivElement,
    getImageDTO: Util['getImageDTO'] = defaultGetImageDTO,
    uploadImage: Util['uploadImage'] = defaultUploadImage
  ) {
    this.stage = stage;
    this.container = container;
    this.util = {
      getImageDTO,
      uploadImage,
      getRegionMaskImage: this._getRegionMaskImage.bind(this),
      getInpaintMaskImage: this._getInpaintMaskImage.bind(this),
      getImageSourceImage: this._getImageSourceImage.bind(this),
      getMaskLayerClone: this._getMaskLayerClone.bind(this),
      getCompositeLayerStageClone: this._getCompositeLayerStageClone.bind(this),
      getGenerationMode: this._getGenerationMode.bind(this),
    };
    this._konvaApi = null;
    this._preview = null;
    this._background = null;
    this._stateApi = null;
    this.adapters = new Map();
  }

  add(entity: CanvasEntity, konvaLayer: Konva.Layer, konvaObjectGroup: Konva.Group): KonvaEntityAdapter {
    const adapter = new KonvaEntityAdapter(entity, konvaLayer, konvaObjectGroup, this);
    this.adapters.set(adapter.id, adapter);
    return adapter;
  }

  get(id: string): KonvaEntityAdapter | undefined {
    return this.adapters.get(id);
  }

  getAll(type?: CanvasEntity['type']): KonvaEntityAdapter[] {
    if (type) {
      return Array.from(this.adapters.values()).filter((adapter) => adapter.entityType === type);
    } else {
      return Array.from(this.adapters.values());
    }
  }

  destroy(id: string): boolean {
    const adapter = this.get(id);
    if (!adapter) {
      return false;
    }
    adapter.konvaLayer.destroy();
    return this.adapters.delete(id);
  }

  set konvaApi(konvaApi: KonvaApi) {
    this._konvaApi = konvaApi;
  }

  get konvaApi(): KonvaApi {
    assert(this._konvaApi !== null, 'Konva API has not been set');
    return this._konvaApi;
  }

  set preview(preview: PreviewLayer) {
    this._preview = preview;
  }

  get preview(): PreviewLayer {
    assert(this._preview !== null, 'Konva preview layer has not been set');
    return this._preview;
  }

  set background(background: BackgroundLayer) {
    this._background = background;
  }

  get background(): BackgroundLayer {
    assert(this._background !== null, 'Konva background layer has not been set');
    return this._background;
  }

  set stateApi(stateApi: StateApi) {
    this._stateApi = stateApi;
  }

  get stateApi(): StateApi {
    assert(this._stateApi !== null, 'State API has not been set');
    return this._stateApi;
  }

  _getMaskLayerClone(arg: { id: string }): Konva.Layer {
    const { id } = arg;
    const adapter = this.get(id);
    assert(adapter, `Adapter for entity ${id} not found`);

    const layerClone = adapter.konvaLayer.clone();
    const objectGroupClone = adapter.konvaObjectGroup.clone();

    layerClone.destroyChildren();
    layerClone.add(objectGroupClone);

    objectGroupClone.opacity(1);
    objectGroupClone.cache();

    return layerClone;
  }

  _getCompositeLayerStageClone(): Konva.Stage {
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

  _getGenerationMode(): GenerationMode {
    const { x, y, width, height } = this.stateApi.getBbox();
    const inpaintMaskLayer = this.util.getMaskLayerClone({ id: 'inpaint_mask' });
    const inpaintMaskImageData = konvaNodeToImageData(inpaintMaskLayer, { x, y, width, height });
    const inpaintMaskTransparency = getImageDataTransparency(inpaintMaskImageData);
    const compositeLayer = this.util.getCompositeLayerStageClone();
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

  async _getRegionMaskImage(arg: { id: string; bbox?: Rect; preview?: boolean }): Promise<ImageDTO> {
    const { id, bbox, preview = false } = arg;
    const region = this.stateApi.getRegionsState().entities.find((entity) => entity.id === id);
    assert(region, `Region entity state with id ${id} not found`);

    if (region.imageCache) {
      const imageDTO = await this.util.getImageDTO(region.imageCache.name);
      if (imageDTO) {
        return imageDTO;
      }
    }

    const layerClone = this.util.getMaskLayerClone({ id });
    const blob = await konvaNodeToBlob(layerClone, bbox);

    if (preview) {
      previewBlob(blob, `region ${region.id} mask`);
    }

    layerClone.destroy();

    const imageDTO = await this.util.uploadImage(blob, `${region.id}_mask.png`, 'mask', true);
    this.stateApi.onRegionMaskImageCached(region.id, imageDTO);
    return imageDTO;
  }

  async _getInpaintMaskImage(arg: { bbox?: Rect; preview?: boolean }): Promise<ImageDTO> {
    const { bbox, preview = false } = arg;
    const inpaintMask = this.stateApi.getInpaintMaskState();

    if (inpaintMask.imageCache) {
      const imageDTO = await this.util.getImageDTO(inpaintMask.imageCache.name);
      if (imageDTO) {
        return imageDTO;
      }
    }

    const layerClone = this.util.getMaskLayerClone({ id: inpaintMask.id });
    const blob = await konvaNodeToBlob(layerClone, bbox);

    if (preview) {
      previewBlob(blob, 'inpaint mask');
    }

    layerClone.destroy();

    const imageDTO = await this.util.uploadImage(blob, 'inpaint_mask.png', 'mask', true);
    this.stateApi.onInpaintMaskImageCached(imageDTO);
    return imageDTO;
  }

  async _getImageSourceImage(arg: { bbox?: Rect; preview?: boolean }): Promise<ImageDTO> {
    const { bbox, preview = false } = arg;
    const { imageCache } = this.stateApi.getLayersState();
    if (imageCache) {
      const imageDTO = await this.util.getImageDTO(imageCache.name);
      if (imageDTO) {
        return imageDTO;
      }
    }

    const stageClone = this.util.getCompositeLayerStageClone();

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

export class KonvaEntityAdapter {
  id: string;
  entityType: CanvasEntity['type'];
  konvaLayer: Konva.Layer; // Every entity is associated with a konva layer
  konvaObjectGroup: Konva.Group; // Every entity's nodes are part of an object group
  objectRecords: Map<string, ObjectRecord>;
  manager: KonvaNodeManager;

  constructor(entity: CanvasEntity, konvaLayer: Konva.Layer, konvaObjectGroup: Konva.Group, manager: KonvaNodeManager) {
    this.id = entity.id;
    this.entityType = entity.type;
    this.konvaLayer = konvaLayer;
    this.konvaObjectGroup = konvaObjectGroup;
    this.objectRecords = new Map();
    this.manager = manager;
    this.konvaLayer.add(this.konvaObjectGroup);
    this.manager.stage.add(this.konvaLayer);
  }

  add<T extends ObjectRecord>(objectRecord: T): T {
    this.objectRecords.set(objectRecord.id, objectRecord);
    if (objectRecord.type === 'brush_line' || objectRecord.type === 'eraser_line') {
      objectRecord.konvaLineGroup.add(objectRecord.konvaLine);
      this.konvaObjectGroup.add(objectRecord.konvaLineGroup);
    } else if (objectRecord.type === 'rect_shape') {
      this.konvaObjectGroup.add(objectRecord.konvaRect);
    } else if (objectRecord.type === 'image') {
      objectRecord.konvaPlaceholderGroup.add(objectRecord.konvaPlaceholderRect);
      objectRecord.konvaPlaceholderGroup.add(objectRecord.konvaPlaceholderText);
      objectRecord.konvaImageGroup.add(objectRecord.konvaPlaceholderGroup);
      this.konvaObjectGroup.add(objectRecord.konvaImageGroup);
    }
    return objectRecord;
  }

  get<T extends ObjectRecord>(id: string): T | undefined {
    return this.objectRecords.get(id) as T | undefined;
  }

  getAll<T extends ObjectRecord>(): T[] {
    return Array.from(this.objectRecords.values()) as T[];
  }

  destroy(id: string): boolean {
    const record = this.get(id);
    if (!record) {
      return false;
    }
    if (record.type === 'brush_line' || record.type === 'eraser_line') {
      record.konvaLineGroup.destroy();
    } else if (record.type === 'rect_shape') {
      record.konvaRect.destroy();
    } else if (record.type === 'image') {
      record.konvaImageGroup.destroy();
    }
    return this.objectRecords.delete(id);
  }
}
