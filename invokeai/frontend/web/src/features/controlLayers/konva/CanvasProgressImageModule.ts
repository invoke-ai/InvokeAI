import { Mutex } from 'async-mutex';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId, loadImage } from 'features/controlLayers/konva/util';
import { selectShowProgressOnCanvas } from 'features/controlLayers/store/canvasSettingsSlice';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasProgressImageModule extends CanvasModuleBase {
  readonly type = 'progress_image';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  progressImageId: string | null = null;
  konva: {
    group: Konva.Group;
    image: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  };
  isLoading: boolean = false;
  isError: boolean = false;
  imageElement: HTMLImageElement | null = null;

  subscriptions = new Set<() => void>();

  mutex: Mutex = new Mutex();

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating progress image module');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      image: null,
    };

    this.subscriptions.add(this.manager.stateApi.$lastCanvasProgressEvent.listen(this.render));
    this.subscriptions.add(this.manager.stagingArea.$shouldShowStagedImage.listen(this.render));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectShowProgressOnCanvas, this.render));
  }

  getNodes = () => {
    return [this.konva.group];
  };

  render = async () => {
    const release = await this.mutex.acquire();

    const event = this.manager.stateApi.$lastCanvasProgressEvent.get();
    const showProgressOnCanvas = this.manager.stateApi.runSelector(selectShowProgressOnCanvas);

    if (!event || !showProgressOnCanvas) {
      this.konva.group.visible(false);
      this.imageElement = null;
      this.isLoading = false;
      this.isError = false;
      release();
      return;
    }

    this.isLoading = true;

    const { x, y, width, height } = this.manager.stateApi.getBbox().rect;
    const { dataURL } = event.progress_image;
    try {
      this.imageElement = await loadImage(dataURL);
      if (this.konva.image) {
        this.konva.image.setAttrs({
          image: this.imageElement,
          x,
          y,
          width,
          height,
        });
      } else {
        this.konva.image = new Konva.Image({
          name: `${this.type}:image`,
          listening: false,
          image: this.imageElement,
          x,
          y,
          width,
          height,
        });
        this.konva.group.add(this.konva.image);
      }
      // Should not be visible if the user has disabled showing staging images
      this.konva.group.visible(this.manager.stagingArea.$shouldShowStagedImage.get());
    } catch {
      this.isError = true;
    } finally {
      this.isLoading = false;
      release();
    }
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.group.destroy();
  };
}
