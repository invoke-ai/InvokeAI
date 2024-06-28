import { KonvaImage } from 'features/controlLayers/konva/renderers/objects';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { assert } from 'tsafe';

export class CanvasStagingArea {
  group: Konva.Group;
  image: KonvaImage | null;

  constructor() {
    this.group = new Konva.Group({ listening: false });
    this.image = null;
  }

  async render(stagingArea: CanvasV2State['stagingArea'], shouldShowStagedImage: boolean) {
    if (!stagingArea || stagingArea.selectedImageIndex === null) {
      if (this.image) {
        this.image.destroy();
        this.image = null;
      }
      return;
    }

    if (stagingArea.selectedImageIndex !== null) {
      const imageDTO = stagingArea.images[stagingArea.selectedImageIndex];
      assert(imageDTO, 'Image must exist');
      if (this.image) {
        if (!this.image.isLoading && !this.image.isError && this.image.imageName !== imageDTO.image_name) {
          await this.image.updateImageSource(imageDTO.image_name);
        }
        this.image.konvaImageGroup.visible(shouldShowStagedImage);
      } else {
        const { image_name, width, height } = imageDTO;
        this.image = new KonvaImage({
          imageObject: {
            id: 'staging-area-image',
            type: 'image',
            x: stagingArea.bbox.x,
            y: stagingArea.bbox.y,
            width,
            height,
            filters: [],
            image: {
              name: image_name,
              width,
              height,
            },
          },
        });
        this.group.add(this.image.konvaImageGroup);
        await this.image.updateImageSource(imageDTO.image_name);
        this.image.konvaImageGroup.visible(shouldShowStagedImage);
      }
    }
  }
}
