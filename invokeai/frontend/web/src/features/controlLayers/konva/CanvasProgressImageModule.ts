import { Mutex } from 'async-mutex';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId, loadImage } from 'features/controlLayers/konva/util';
import { selectShowProgressOnCanvas } from 'features/controlLayers/store/canvasSettingsSlice';
import Konva from 'konva';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { selectCanvasQueueCounts } from 'services/api/endpoints/queue';
import type { S } from 'services/api/types';
import type { O } from 'ts-toolbelt';

type ProgressEventWithImage = O.NonNullable<S['InvocationProgressEvent'], 'image'>;
const isProgressEventWithImage = (val: S['InvocationProgressEvent']): val is ProgressEventWithImage =>
  Boolean(val.image);

export class CanvasProgressImageModule extends CanvasModuleBase {
  readonly type = 'progress_image';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  konva: {
    group: Konva.Group;
    image: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  };
  isLoading: boolean = false;
  isError: boolean = false;
  imageElement: HTMLImageElement | null = null;

  subscriptions = new Set<() => void>();
  $lastProgressEvent = atom<ProgressEventWithImage | null>(null);
  hasActiveGeneration: boolean = false;
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

    this.subscriptions.add(this.manager.stagingArea.$shouldShowStagedImage.listen(this.render));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectShowProgressOnCanvas, this.render));
    this.subscriptions.add(this.setSocketEventListeners());
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectCanvasQueueCounts, ({ data }) => {
        if (data && (data.in_progress > 0 || data.pending > 0)) {
          this.hasActiveGeneration = true;
        } else {
          this.hasActiveGeneration = false;
          this.$lastProgressEvent.set(null);
        }
      })
    );
    this.subscriptions.add(this.$lastProgressEvent.listen(this.render));
  }

  setSocketEventListeners = (): (() => void) => {
    const progressListener = (data: S['InvocationProgressEvent']) => {
      if (data.destination !== 'canvas') {
        return;
      }
      if (!isProgressEventWithImage(data)) {
        return;
      }
      if (!this.hasActiveGeneration) {
        return;
      }
      this.$lastProgressEvent.set(data);
    };

    const clearProgress = () => {
      this.$lastProgressEvent.set(null);
    };

    this.manager.socket.on('invocation_progress', progressListener);
    this.manager.socket.on('connect', clearProgress);
    this.manager.socket.on('connect_error', clearProgress);
    this.manager.socket.on('disconnect', clearProgress);

    return () => {
      this.manager.socket.off('invocation_progress', progressListener);
      this.manager.socket.off('connect', clearProgress);
      this.manager.socket.off('connect_error', clearProgress);
      this.manager.socket.off('disconnect', clearProgress);
    };
  };

  getNodes = () => {
    return [this.konva.group];
  };

  render = async () => {
    const release = await this.mutex.acquire();

    const event = this.$lastProgressEvent.get();
    const showProgressOnCanvas = this.manager.stateApi.runSelector(selectShowProgressOnCanvas);

    if (!event || !showProgressOnCanvas) {
      this.konva.group.visible(false);
      this.konva.image?.destroy();
      this.konva.image = null;
      this.imageElement = null;
      this.isLoading = false;
      this.isError = false;
      release();
      return;
    }

    this.isLoading = true;

    const { x, y, width, height } = this.manager.stateApi.getBbox().rect;
    try {
      this.imageElement = await loadImage(event.image.dataURL);
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
          perfectDrawEnabled: false,
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
