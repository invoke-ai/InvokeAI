import { CanvasEntity } from 'features/controlLayers/konva/CanvasEntity';
import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { StagingAreaImage } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasStagingArea extends CanvasEntity {
  static NAME_PREFIX = 'staging-area';
  static GROUP_NAME = `${CanvasStagingArea.NAME_PREFIX}_group`;

  type = 'staging_area';
  konva: { group: Konva.Group };

  image: CanvasImage | null;
  selectedImage: StagingAreaImage | null;

  constructor(manager: CanvasManager) {
    super('staging-area', manager);
    this.konva = { group: new Konva.Group({ name: CanvasStagingArea.GROUP_NAME, listening: false }) };
    this.image = null;
    this.selectedImage = null;
  }

  async render() {
    const session = this._manager.stateApi.getSession();
    const bboxRect = this._manager.stateApi.getBbox().rect;
    const shouldShowStagedImage = this._manager.stateApi.getShouldShowStagedImage();

    this.selectedImage = session.stagedImages[session.selectedStagedImageIndex] ?? null;

    if (this.selectedImage) {
      const { imageDTO, offsetX, offsetY } = this.selectedImage;

      if (!this.image) {
        const { image_name, width, height } = imageDTO;
        this.image = new CanvasImage(
          {
            id: 'staging-area-image',
            type: 'image',
            x: 0,
            y: 0,
            width,
            height,
            filters: [],
            image: {
              name: image_name,
              width,
              height,
            },
          },
          this
        );
        this.konva.group.add(this.image.konva.group);
      }

      if (!this.image.isLoading && !this.image.isError && this.image.imageName !== imageDTO.image_name) {
        this.image.konva.image?.width(imageDTO.width);
        this.image.konva.image?.height(imageDTO.height);
        this.image.konva.group.x(bboxRect.x + offsetX);
        this.image.konva.group.y(bboxRect.y + offsetY);
        await this.image.updateImageSource(imageDTO.image_name);
        this._manager.stateApi.resetLastProgressEvent();
      }
      this.image.konva.group.visible(shouldShowStagedImage);
    } else {
      this.image?.konva.group.visible(false);
    }
  }

  repr() {
    return {
      id: this.id,
      type: this.type,
      selectedImage: this.selectedImage,
    };
  }
}
