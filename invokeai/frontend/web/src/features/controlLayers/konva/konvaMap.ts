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
  konvaImage: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  konvaGroup: Konva.Group;
};

type Entry = BrushLineEntry | EraserLineEntry | RectShapeEntry | ImageEntry;

export class EntityToKonvaMap {
  mappings: Record<string, EntityToKonvaMapping>;

  constructor() {
    this.mappings = {};
  }

  addMapping(id: string, konvaLayer: Konva.Layer, konvaObjectGroup: Konva.Group): EntityToKonvaMapping {
    const mapping = new EntityToKonvaMapping(id, konvaLayer, konvaObjectGroup);
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

  constructor(id: string, konvaLayer: Konva.Layer, konvaObjectGroup: Konva.Group) {
    this.id = id;
    this.konvaLayer = konvaLayer;
    this.konvaObjectGroup = konvaObjectGroup;
    this.konvaNodeEntries = {};
  }

  addEntry<T extends Entry>(entry: T): T {
    this.konvaNodeEntries[entry.id] = entry;
    return entry;
  }

  getEntry<T extends Entry>(id: string): T | undefined {
    return this.konvaNodeEntries[id] as T | undefined;
  }

  getEntries(): Entry[] {
    return Object.values(this.konvaNodeEntries);
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
      entry.konvaGroup.destroy();
    }
    delete this.konvaNodeEntries[id];
  }
}
