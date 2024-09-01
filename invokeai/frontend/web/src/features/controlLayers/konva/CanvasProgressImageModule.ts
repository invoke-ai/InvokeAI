import { Mutex } from 'async-mutex';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId, loadImage } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';
import type { S } from 'services/api/types';

export class CanvasProgressImageModule extends CanvasModuleBase {
  readonly type = 'progress_image';

  id: string;
  path: string[];
  parent: CanvasManager;
  manager: CanvasManager;
  log: Logger;

  progressImageId: string | null = null;
  konva: {
    group: Konva.Group;
    image: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  };
  isLoading: boolean = false;
  isError: boolean = false;
  imageElement: HTMLImageElement | null = null;

  lastProgressEvent: S['InvocationDenoiseProgressEvent'] | null = null;

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
    this.log.debug('Destroying module');
    this.konva.group.destroy();
  };
}
