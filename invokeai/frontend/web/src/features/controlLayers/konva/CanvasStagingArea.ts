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
    const shouldShowStagedImage = this.manager.stateApi.getShouldShowStagedImage();

    this.selectedImage = session.stagedImages[session.selectedStagedImageIndex] ?? null;

    if (this.selectedImage) {
      if (this.image) {
        if (
          !this.image.isLoading &&
          !this.image.isError &&
          this.image.imageName !== this.selectedImage.imageDTO.image_name
        ) {
          await this.image.updateImageSource(this.selectedImage.imageDTO.image_name);
        }
        this.image.konvaImageGroup.x(this.selectedImage.rect.x);
        this.image.konvaImageGroup.y(this.selectedImage.rect.y);
        this.image.konvaImageGroup.visible(shouldShowStagedImage);
      } else {
        const { image_name } = this.selectedImage.imageDTO;
        const { x, y, width, height } = this.selectedImage.rect;
        this.image = new CanvasImage(
          {
            id: 'staging-area-image',
            type: 'image',
            x,
            y,
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
              if (this.selectedImage) {
                konvaImage.width(this.selectedImage.rect.width);
                konvaImage.height(this.selectedImage.rect.height);
              }
              this.manager.stateApi.resetLastProgressEvent();
              this.image?.konvaImageGroup.visible(shouldShowStagedImage);
            },
          }
        );
        this.group.add(this.image.konvaImageGroup);
        await this.image.updateImageSource(this.selectedImage.imageDTO.image_name);
        this.image.konvaImageGroup.visible(shouldShowStagedImage);
      }
    } else {
      this.image?.konvaImageGroup.visible(false);
    }
  }
}
