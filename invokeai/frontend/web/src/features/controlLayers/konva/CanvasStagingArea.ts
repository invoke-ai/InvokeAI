import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { GetLoggingContext, StagingAreaImage } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasStagingArea {
  static TYPE = 'staging_area';
  static GROUP_NAME = `${CanvasStagingArea.TYPE}_group`;

  id: string;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  konva: { group: Konva.Group };

  image: CanvasImage | null;
  selectedImage: StagingAreaImage | null;

  constructor(manager: CanvasManager) {
    this.id = getPrefixedId(CanvasStagingArea.TYPE);
    this.manager = manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug('Creating staging area');

    this.konva = { group: new Konva.Group({ name: CanvasStagingArea.GROUP_NAME, listening: false }) };
    this.image = null;
    this.selectedImage = null;
  }

  render = async () => {
    const session = this.manager.stateApi.getSession();
    const bboxRect = this.manager.stateApi.getBbox().rect;
    const shouldShowStagedImage = this.manager.stateApi.getShouldShowStagedImage();

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
        this.manager.stateApi.resetLastProgressEvent();
      }
      this.image.konva.group.visible(shouldShowStagedImage);
    } else {
      this.image?.konva.group.visible(false);
    }
  };

  repr = () => {
    return {
      id: this.id,
      type: CanvasStagingArea.TYPE,
      selectedImage: this.selectedImage,
    };
  };
}
