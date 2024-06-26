import { LightnessToAlphaFilter } from 'features/controlLayers/konva/filters';
import { getObjectGroupId } from 'features/controlLayers/konva/naming';
import type { ControlAdapterEntity } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { isEqual } from 'lodash-es';
import { v4 as uuidv4 } from 'uuid';

import { KonvaImage } from './objects';

export class CanvasControlAdapter {
  id: string;
  konvaLayer: Konva.Layer;
  konvaObjectGroup: Konva.Group;
  konvaImageObject: KonvaImage | null;

  constructor(entity: ControlAdapterEntity) {
    const { id } = entity;
    this.id = id;
    this.konvaLayer = new Konva.Layer({
      id,
      imageSmoothingEnabled: false,
      listening: false,
    });
    this.konvaObjectGroup = new Konva.Group({
      id: getObjectGroupId(this.konvaLayer.id(), uuidv4()),
      listening: false,
    });
    this.konvaLayer.add(this.konvaObjectGroup);
    this.konvaImageObject = null;
  }

  async render(entity: ControlAdapterEntity) {
    const imageObject = entity.processedImageObject ?? entity.imageObject;
    if (!imageObject) {
      if (this.konvaImageObject) {
        this.konvaImageObject.destroy();
      }
      return;
    }

    const opacity = entity.opacity;
    const visible = entity.isEnabled;
    const filters = entity.filter === 'LightnessToAlphaFilter' ? [LightnessToAlphaFilter] : [];

    if (!this.konvaImageObject) {
      this.konvaImageObject = await new KonvaImage({
        imageObject,
        onLoad: (konvaImage) => {
          konvaImage.filters(filters);
          konvaImage.cache();
          konvaImage.opacity(opacity);
          konvaImage.visible(visible);
        },
      });
      this.konvaObjectGroup.add(this.konvaImageObject.konvaImageGroup);
    }
    if (this.konvaImageObject.isLoading || this.konvaImageObject.isError) {
      return;
    }
    if (this.konvaImageObject.imageName !== imageObject.image.name) {
      this.konvaImageObject.updateImageSource(imageObject.image.name);
    }
    if (this.konvaImageObject.konvaImage) {
      if (!isEqual(this.konvaImageObject.konvaImage.filters(), filters)) {
        this.konvaImageObject.konvaImage.filters(filters);
        this.konvaImageObject.konvaImage.cache();
      }
      this.konvaImageObject.konvaImage.opacity(opacity);
      this.konvaImageObject.konvaImage.visible(visible);
    }
  }

  destroy(): void {
    this.konvaLayer.destroy();
  }
}
