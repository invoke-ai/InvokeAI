import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { StagingAreaImage } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasStagingArea {
  group: Konva.Group;
  image: CanvasImage | null;
  selectedImage: StagingAreaImage | null;
  manager: CanvasManager;

  constructor(manager: CanvasManager) {
    this.manager = manager;
    this.group = new Konva.Group({ listening: false });
    this.image = null;
    this.selectedImage = null;
  }

  async render() {
    const session = this.manager.stateApi.getSession();
    const bboxRect = this.manager.stateApi.getBbox().rect;
    const shouldShowStagedImage = this.manager.stateApi.getShouldShowStagedImage();

    this.selectedImage = session.stagedImages[session.selectedStagedImageIndex] ?? null;

    if (this.selectedImage) {
      const { imageDTO, offsetX, offsetY } = this.selectedImage;

      if (!this.image) {
        const { image_name, width, height } = imageDTO;
        this.image = new CanvasImage({
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
        });
        this.group.add(this.image.konvaImageGroup);
      }

      if (!this.image.isLoading && !this.image.isError && this.image.imageName !== imageDTO.image_name) {
        this.image.konvaImage?.width(imageDTO.width);
        this.image.konvaImage?.height(imageDTO.height);
        this.image.konvaImageGroup.x(bboxRect.x + offsetX);
        this.image.konvaImageGroup.y(bboxRect.y + offsetY);
        await this.image.updateImageSource(imageDTO.image_name);
        this.manager.stateApi.resetLastProgressEvent();
      }
      this.image.konvaImageGroup.visible(shouldShowStagedImage);
    } else {
      this.image?.konvaImageGroup.visible(false);
    }
  }
}
