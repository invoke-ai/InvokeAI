import { Mutex } from 'async-mutex';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasPreview } from 'features/controlLayers/konva/CanvasPreview';
import { getPrefixedId, loadImage } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { InvocationDenoiseProgressEvent } from 'services/events/types';

export class CanvasProgressImage {
  static NAME_PREFIX = 'progress-image';
  static GROUP_NAME = `${CanvasProgressImage.NAME_PREFIX}_group`;
  static IMAGE_NAME = `${CanvasProgressImage.NAME_PREFIX}_image`;

  id: string;
  parent: CanvasPreview;
  manager: CanvasManager;

  /**
   * A set of subscriptions that should be cleaned up when the transformer is destroyed.
   */
  subscriptions: Set<() => void> = new Set();

  progressImageId: string | null = null;
  konva: {
    group: Konva.Group;
    image: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  };
  isLoading: boolean = false;
  isError: boolean = false;
  imageElement: HTMLImageElement | null = null;

  lastProgressEvent: InvocationDenoiseProgressEvent | null = null;

  mutex: Mutex = new Mutex();

  constructor(parent: CanvasPreview) {
    this.id = getPrefixedId(CanvasProgressImage.NAME_PREFIX);
    this.parent = parent;
    this.manager = parent.manager;
    this.konva = {
      group: new Konva.Group({ name: CanvasProgressImage.GROUP_NAME, listening: false }),
      image: null,
    };

    this.manager.stateApi.$lastProgressEvent.listen((event) => {
      this.lastProgressEvent = event;
      this.render();
    });
  }

  getNodes = () => {
    return [this.konva.group];
  };

  render = async () => {
    const release = await this.mutex.acquire();

    if (!this.lastProgressEvent) {
      this.konva.group.visible(false);
      this.imageElement = null;
      this.isLoading = false;
      this.isError = false;
      release();
      return;
    }

    const { isStaging } = this.manager.stateApi.getSession();

    if (!isStaging) {
      release();
      return;
    }

    this.isLoading = true;

    const { x, y, width, height } = this.manager.stateApi.getBbox().rect;
    const { dataURL } = this.lastProgressEvent.progress_image;
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
          name: CanvasProgressImage.IMAGE_NAME,
          listening: false,
          image: this.imageElement,
          x,
          y,
          width,
          height,
        });
        this.konva.group.add(this.konva.image);
      }
      this.konva.group.visible(true);
    } catch {
      this.isError = true;
    } finally {
      this.isLoading = false;
      release();
    }
  };

  destroy = () => {
    for (const unsubscribe of this.subscriptions) {
      unsubscribe();
    }
    this.konva.group.destroy();
  };
}
