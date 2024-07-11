import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getObjectGroupId } from 'features/controlLayers/konva/naming';
import type { InitialImageEntity } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { v4 as uuidv4 } from 'uuid';

export class CanvasInitialImage {
  id = 'initial_image';
  manager: CanvasManager;
  layer: Konva.Layer;
  group: Konva.Group;
  objectsGroup: Konva.Group;
  image: CanvasImage | null;
  private initialImageState: InitialImageEntity;

  constructor(initialImageState: InitialImageEntity, manager: CanvasManager) {
    this.manager = manager;
    this.layer = new Konva.Layer({
      id: this.id,
      imageSmoothingEnabled: true,
      listening: false,
    });
    this.group = new Konva.Group({
      id: getObjectGroupId(this.layer.id(), uuidv4()),
      listening: false,
    });
    this.objectsGroup = new Konva.Group({ listening: false });
    this.group.add(this.objectsGroup);
    this.layer.add(this.group);

    this.image = null;
    this.initialImageState = initialImageState;
  }

  async render(initialImageState: InitialImageEntity) {
    this.initialImageState = initialImageState;

    if (!this.initialImageState.imageObject) {
      this.layer.visible(false);
      return;
    }

    if (!this.image) {
      this.image = await new CanvasImage(this.initialImageState.imageObject, {});
      this.objectsGroup.add(this.image.konvaImageGroup);
      await this.image.update(this.initialImageState.imageObject, true);
    } else if (!this.image.isLoading && !this.image.isError) {
      await this.image.update(this.initialImageState.imageObject);
    }

    if (this.initialImageState && this.initialImageState.isEnabled && !this.image?.isLoading && !this.image?.isError) {
      this.layer.visible(true);
    } else {
      this.layer.visible(false);
    }
  }

  destroy(): void {
    this.layer.destroy();
  }
}
