import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import { LightnessToAlphaFilter } from 'features/controlLayers/konva/filters';
import { getObjectGroupId } from 'features/controlLayers/konva/naming';
import type { ControlAdapterEntity } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { isEqual } from 'lodash-es';
import { v4 as uuidv4 } from 'uuid';

export class CanvasControlAdapter {
  id: string;
  layer: Konva.Layer;
  group: Konva.Group;
  image: CanvasImage | null;

  constructor(entity: ControlAdapterEntity) {
    const { id } = entity;
    this.id = id;
    this.layer = new Konva.Layer({
      id,
      imageSmoothingEnabled: false,
      listening: false,
    });
    this.group = new Konva.Group({
      id: getObjectGroupId(this.layer.id(), uuidv4()),
      listening: false,
    });
    this.layer.add(this.group);
    this.image = null;
  }

  async render(entity: ControlAdapterEntity) {
    const imageObject = entity.processedImageObject ?? entity.imageObject;
    if (!imageObject) {
      if (this.image) {
        this.image.konvaImageGroup.visible(false);
      }
      return;
    }

    const opacity = entity.opacity;
    const visible = entity.isEnabled;
    const filters = entity.filter === 'LightnessToAlphaFilter' ? [LightnessToAlphaFilter] : [];

    if (!this.image) {
      this.image = await new CanvasImage(imageObject, {
        onLoad: (konvaImage) => {
          konvaImage.filters(filters);
          konvaImage.cache();
          konvaImage.opacity(opacity);
          konvaImage.visible(visible);
        },
      });
      this.group.add(this.image.konvaImageGroup);
    }
    if (this.image.isLoading || this.image.isError) {
      return;
    }
    if (this.image.imageName !== imageObject.image.name) {
      this.image.updateImageSource(imageObject.image.name);
    }
    if (this.image.konvaImage) {
      if (!isEqual(this.image.konvaImage.filters(), filters)) {
        this.image.konvaImage.filters(filters);
        this.image.konvaImage.cache();
      }
      this.image.konvaImage.opacity(opacity);
      this.image.konvaImage.visible(visible);
    }
  }

  destroy(): void {
    this.layer.destroy();
  }
}
