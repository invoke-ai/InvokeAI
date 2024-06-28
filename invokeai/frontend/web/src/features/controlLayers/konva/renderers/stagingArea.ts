import { KonvaImage, KonvaProgressImage } from 'features/controlLayers/konva/renderers/objects';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { InvocationDenoiseProgressEvent } from 'services/events/types';
import { assert } from 'tsafe';

export class CanvasStagingArea {
  group: Konva.Group;
  image: KonvaImage | null;
  progressImage: KonvaProgressImage | null;

  constructor() {
    this.group = new Konva.Group({ listening: false });
    this.image = null;
    this.progressImage = null;
  }

  async render(
    stagingArea: CanvasV2State['stagingArea'],
    shouldShowStagedImage: boolean,
    lastProgressEvent: InvocationDenoiseProgressEvent | null
  ) {
    if (stagingArea && lastProgressEvent) {
      const { invocation, step, progress_image } = lastProgressEvent;
      const { dataURL } = progress_image;
      const { x, y, width, height } = stagingArea.bbox;
      const progressImageId = `${invocation.id}_${step}`;
      if (this.progressImage) {
        if (
          !this.progressImage.isLoading &&
          !this.progressImage.isError &&
          this.progressImage.progressImageId !== progressImageId
        ) {
          await this.progressImage.updateImageSource(progressImageId, dataURL, x, y, width, height);
          this.image?.konvaImageGroup.visible(false);
          this.progressImage.konvaImageGroup.visible(true);
        }
      } else {
        this.progressImage = new KonvaProgressImage({ id: 'progress-image' });
        this.group.add(this.progressImage.konvaImageGroup);
        await this.progressImage.updateImageSource(progressImageId, dataURL, x, y, width, height);
        this.image?.konvaImageGroup.visible(false);
        this.progressImage.konvaImageGroup.visible(true);
      }
    } else if (stagingArea && stagingArea.selectedImageIndex !== null) {
      const imageDTO = stagingArea.images[stagingArea.selectedImageIndex];
      assert(imageDTO, 'Image must exist');
      if (this.image) {
        if (!this.image.isLoading && !this.image.isError && this.image.imageName !== imageDTO.image_name) {
          await this.image.updateImageSource(imageDTO.image_name);
        }
        this.image.konvaImageGroup.x(stagingArea.bbox.x);
        this.image.konvaImageGroup.y(stagingArea.bbox.y);
        this.image.konvaImageGroup.visible(shouldShowStagedImage);
        this.progressImage?.konvaImageGroup.visible(false);
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
        this.progressImage?.konvaImageGroup.visible(false);
      }
    } else {
      if (this.image) {
        this.image.konvaImageGroup.visible(false);
      }
      if (this.progressImage) {
        this.progressImage.konvaImageGroup.visible(false);
      }
    }
  }
}
