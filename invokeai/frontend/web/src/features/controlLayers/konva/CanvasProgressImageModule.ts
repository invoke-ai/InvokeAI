import { Mutex } from 'async-mutex';
import type { JSONObject } from 'common/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasPreviewModule } from 'features/controlLayers/konva/CanvasPreviewModule';
import { getPrefixedId, loadImage } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';
import type { InvocationDenoiseProgressEvent } from 'services/events/types';

export class CanvasProgressImageModule {
  readonly type = 'progress_image';

  id: string;
  path: string[];
  parent: CanvasPreviewModule;
  manager: CanvasManager;
  log: Logger;

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

  constructor(parent: CanvasPreviewModule) {
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.trace('Creating progress image');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      image: null,
    };

    this.manager.stateApi.$lastCanvasProgressEvent.listen((event) => {
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
      this.konva.group.visible(true);
    } catch {
      this.isError = true;
    } finally {
      this.isLoading = false;
      release();
    }
  };

  destroy = () => {
    this.log.trace('Destroying progress image');
    for (const unsubscribe of this.subscriptions) {
      unsubscribe();
    }
    this.konva.group.destroy();
  };

  getLoggingContext = (): JSONObject => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
