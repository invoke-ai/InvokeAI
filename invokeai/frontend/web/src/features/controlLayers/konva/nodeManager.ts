import type { BrushLine, EraserLine, ImageObject, RectShape } from 'features/controlLayers/store/types';
import type Konva from 'konva';

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

export class KonvaNodeManager {
  stage: Konva.Stage;
  adapters: Map<string, EntityKonvaAdapter>;

  constructor(stage: Konva.Stage) {
    this.stage = stage;
    this.adapters = new Map();
  }

  add(id: string, konvaLayer: Konva.Layer, konvaObjectGroup: Konva.Group): EntityKonvaAdapter {
    const adapter = new EntityKonvaAdapter(id, konvaLayer, konvaObjectGroup, this);
    this.adapters.set(id, adapter);
    return adapter;
  }

  get(id: string): EntityKonvaAdapter | undefined {
    return this.adapters.get(id);
  }

  getAll(): EntityKonvaAdapter[] {
    return Array.from(this.adapters.values());
  }

  destroy(id: string): boolean {
    const adapter = this.get(id);
    if (!adapter) {
      return false;
    }
    adapter.konvaLayer.destroy();
    return this.adapters.delete(id);
  }
}

export class EntityKonvaAdapter {
  id: string;
  konvaLayer: Konva.Layer; // Every entity is associated with a konva layer
  konvaObjectGroup: Konva.Group; // Every entity's nodes are part of an object group
  objectRecords: Map<string, ObjectRecord>;
  manager: KonvaNodeManager;

  constructor(id: string, konvaLayer: Konva.Layer, konvaObjectGroup: Konva.Group, manager: KonvaNodeManager) {
    this.id = id;
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
