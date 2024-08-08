import { CanvasImageRenderer } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasPreview } from 'features/controlLayers/konva/CanvasPreview';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { GetLoggingContext, StagingAreaImage } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasStagingArea {
  static TYPE = 'staging_area';
  static GROUP_NAME = `${CanvasStagingArea.TYPE}_group`;

  id: string;
  parent: CanvasPreview;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  konva: { group: Konva.Group };

  image: CanvasImageRenderer | null;
  selectedImage: StagingAreaImage | null;

  /**
   * A set of subscriptions that should be cleaned up when the transformer is destroyed.
   */
  subscriptions: Set<() => void> = new Set();

  constructor(parent: CanvasPreview) {
    this.id = getPrefixedId(CanvasStagingArea.TYPE);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug('Creating staging area');

    this.konva = { group: new Konva.Group({ name: CanvasStagingArea.GROUP_NAME, listening: false }) };
    this.image = null;
    this.selectedImage = null;

    this.subscriptions.add(this.manager.stateApi.$shouldShowStagedImage.listen(this.render));
  }

  render = async () => {
    const session = this.manager.stateApi.getSession();
    const { rect } = this.manager.stateApi.getBbox();
    const shouldShowStagedImage = this.manager.stateApi.$shouldShowStagedImage.get();

    this.selectedImage = session.stagedImages[session.selectedStagedImageIndex] ?? null;
    this.konva.group.position({ x: rect.x, y: rect.y });

    if (this.selectedImage) {
      const { imageDTO, offsetX, offsetY } = this.selectedImage;

      if (!this.image) {
        const { image_name, width, height } = imageDTO;
        this.image = new CanvasImageRenderer(
          {
            id: 'staging-area-image',
            type: 'image',
            filters: [],
            image: {
              image_name: image_name,
              width,
              height,
            },
          },
          this
        );
        this.konva.group.add(this.image.konva.group);
      }

      if (!this.image.isLoading && !this.image.isError) {
        await this.image.updateImageSource(imageDTO.image_name);
        this.manager.stateApi.$lastProgressEvent.set(null);
      }
      this.image.konva.group.visible(shouldShowStagedImage);
    } else {
      this.image?.konva.group.visible(false);
    }
  };

  getNodes = () => {
    return [this.konva.group];
  };

  destroy = () => {
    if (this.image) {
      this.image.destroy();
    }
    for (const unsubscribe of this.subscriptions) {
      unsubscribe();
    }
    for (const node of this.getNodes()) {
      node.destroy();
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
