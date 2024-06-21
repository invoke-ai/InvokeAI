import type { BrushLine, CanvasEntity, EraserLine, ImageObject, RectShape } from 'features/controlLayers/store/types';
import type Konva from 'konva';
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

type KonvaRenderers = {
  renderRegions: () => void;
  renderLayers: () => void;
  renderControlAdapters: () => void;
  renderInpaintMask: () => void;
  renderBbox: () => void;
  renderDocumentOverlay: () => void;
  renderBackground: () => void;
  renderToolPreview: () => void;
  fitDocumentToStage: () => void;
  arrangeEntities: () => void;
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

export class KonvaNodeManager {
  stage: Konva.Stage;
  adapters: Map<string, KonvaEntityAdapter>;
  _background: BackgroundLayer | null;
  _preview: PreviewLayer | null;
  _renderers: KonvaRenderers | null;

  constructor(stage: Konva.Stage) {
    this.stage = stage;
    this._renderers = null;
    this._preview = null;
    this._background = null;
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

  set renderers(renderers: KonvaRenderers) {
    this._renderers = renderers;
  }

  get renderers(): KonvaRenderers {
    assert(this._renderers !== null, 'Konva renderers have not been set');
    return this._renderers;
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
