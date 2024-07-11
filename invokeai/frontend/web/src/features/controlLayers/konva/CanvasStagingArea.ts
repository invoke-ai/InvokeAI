import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import Konva from 'konva';
import type { ImageDTO } from 'services/api/types';

export class CanvasStagingArea {
  group: Konva.Group;
  image: CanvasImage | null;
  imageDTO: ImageDTO | null;
  manager: CanvasManager;

  constructor(manager: CanvasManager) {
    this.manager = manager;
    this.group = new Konva.Group({ listening: false });
    this.image = null;
    this.imageDTO = null;
  }

  async render() {
    const session = this.manager.stateApi.getSession();
    const bboxRect = this.manager.stateApi.getBbox().rect;
    const shouldShowStagedImage = this.manager.stateApi.getShouldShowStagedImage();

    this.imageDTO = session.stagedImages[session.selectedStagedImageIndex] ?? null;

    if (this.imageDTO) {
      if (this.image) {
        if (!this.image.isLoading && !this.image.isError && this.image.imageName !== this.imageDTO.image_name) {
          await this.image.updateImageSource(this.imageDTO.image_name);
        }
        this.image.konvaImageGroup.x(bboxRect.x);
        this.image.konvaImageGroup.y(bboxRect.y);
        this.image.konvaImageGroup.visible(shouldShowStagedImage);
      } else {
        const { image_name, width, height } = this.imageDTO;
        this.image = new CanvasImage(
          {
            id: 'staging-area-image',
            type: 'image',
            x: bboxRect.x,
            y: bboxRect.y,
            width,
            height,
            filters: [],
            image: {
              name: image_name,
              width,
              height,
            },
          },
          {
            onLoad: (konvaImage) => {
              if (this.imageDTO) {
                konvaImage.width(this.imageDTO.width);
                konvaImage.height(this.imageDTO.height);
              }
              this.manager.stateApi.resetLastProgressEvent();
              this.image?.konvaImageGroup.visible(shouldShowStagedImage);
            },
          }
        );
        this.group.add(this.image.konvaImageGroup);
        await this.image.updateImageSource(this.imageDTO.image_name);
        this.image.konvaImageGroup.visible(shouldShowStagedImage);
      }
    } else {
      this.image?.konvaImageGroup.visible(false);
    }
  }
}
