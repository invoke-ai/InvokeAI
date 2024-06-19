import type { BrushLine, EraserLine, ImageObject, RectShape } from 'features/controlLayers/store/types';
import type Konva from 'konva';

export type BrushLineEntry = {
  id: string;
  type: BrushLine['type'];
  konvaLine: Konva.Line;
  konvaLineGroup: Konva.Group;
};

export type EraserLineEntry = {
  id: string;
  type: EraserLine['type'];
  konvaLine: Konva.Line;
  konvaLineGroup: Konva.Group;
};

export type RectShapeEntry = {
  id: string;
  type: RectShape['type'];
  konvaRect: Konva.Rect;
};

export type ImageEntry = {
  id: string;
  type: ImageObject['type'];
  konvaImageGroup: Konva.Group;
  konvaPlaceholderGroup: Konva.Group;
  konvaPlaceholderText: Konva.Text;
  konvaImage: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  isLoading: boolean;
  isError: boolean;
};

type Entry = BrushLineEntry | EraserLineEntry | RectShapeEntry | ImageEntry;

export class EntityToKonvaMap {
  stage: Konva.Stage;
  mappings: Record<string, EntityToKonvaMapping>;

  constructor(stage: Konva.Stage) {
    this.stage = stage;
    this.mappings = {};
  }

  addMapping(id: string, konvaLayer: Konva.Layer, konvaObjectGroup: Konva.Group): EntityToKonvaMapping {
    const mapping = new EntityToKonvaMapping(id, konvaLayer, konvaObjectGroup, this);
    this.mappings[id] = mapping;
    return mapping;
  }

  getMapping(id: string): EntityToKonvaMapping | undefined {
    return this.mappings[id];
  }

  getMappings(): EntityToKonvaMapping[] {
    return Object.values(this.mappings);
  }

  destroyMapping(id: string): void {
    const mapping = this.getMapping(id);
    if (!mapping) {
      return;
    }
    mapping.konvaObjectGroup.destroy();
    delete this.mappings[id];
  }
}

export class EntityToKonvaMapping {
  id: string;
  konvaLayer: Konva.Layer;
  konvaObjectGroup: Konva.Group;
  konvaNodeEntries: Record<string, Entry>;
  map: EntityToKonvaMap;

  constructor(id: string, konvaLayer: Konva.Layer, konvaObjectGroup: Konva.Group, map: EntityToKonvaMap) {
    this.id = id;
    this.konvaLayer = konvaLayer;
    this.konvaObjectGroup = konvaObjectGroup;
    this.konvaNodeEntries = {};
    this.map = map;
  }

  addEntry<T extends Entry>(entry: T): T {
    this.konvaNodeEntries[entry.id] = entry;
    return entry;
  }

  getEntry<T extends Entry>(id: string): T | undefined {
    return this.konvaNodeEntries[id] as T | undefined;
  }

  getEntries<T extends Entry>(): T[] {
    return Object.values(this.konvaNodeEntries) as T[];
  }

  destroyEntry(id: string): void {
    const entry = this.getEntry(id);
    if (!entry) {
      return;
    }
    if (entry.type === 'brush_line' || entry.type === 'eraser_line') {
      entry.konvaLineGroup.destroy();
    } else if (entry.type === 'rect_shape') {
      entry.konvaRect.destroy();
    } else if (entry.type === 'image') {
      entry.konvaImageGroup.destroy();
    }
    delete this.konvaNodeEntries[id];
  }
}
