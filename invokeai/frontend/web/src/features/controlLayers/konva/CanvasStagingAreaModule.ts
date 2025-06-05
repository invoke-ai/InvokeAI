import { Mutex } from 'async-mutex';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasImageState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';

export class CanvasStagingAreaModule extends CanvasModuleBase {
  readonly type = 'staging_area';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  subscriptions: Set<() => void> = new Set();
  konva: { group: Konva.Group };
  image: CanvasObjectImage | null;
  mutex = new Mutex();

  $shouldShowStagedImage = atom<boolean>(true);
  $isStaging = atom(true); //TODO: wire up to queue?

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.konva = { group: new Konva.Group({ name: `${this.type}:group`, listening: false }) };
    this.image = null;

    /**
     * When we change this flag, we need to re-render the staging area, which hides or shows the staged image.
     */
    this.subscriptions.add(
      this.$shouldShowStagedImage.listen(() => {
        this.render();
      })
    );
  }

  initialize = () => {
    this.log.debug('Initializing module');
    this.render();
  };

  getImageFromSrc = (
    { type, data }: { type: 'imageName'; data: string } | { type: 'dataURL'; data: string },
    width: number,
    height: number
  ): CanvasImageState['image'] => {
    if (type === 'imageName') {
      return {
        image_name: data,
        width,
        height,
      };
    } else {
      return {
        dataURL: data,
        width,
        height,
      };
    }
  };

  render = async (imageSrc?: { type: 'imageName'; data: string } | { type: 'dataURL'; data: string }) => {
    const release = await this.mutex.acquire();
    try {
      this.log.trace('Rendering staging area');

      const { x, y, width, height } = this.manager.stateApi.getBbox().rect;
      const shouldShowStagedImage = this.$shouldShowStagedImage.get();

      this.konva.group.position({ x, y });

      if (imageSrc) {
        const image = this.getImageFromSrc(imageSrc, width, height);
        if (!this.image) {
          this.image = new CanvasObjectImage({ id: 'staging-area-image', type: 'image', image }, this);
          await this.image.update(this.image.state, true);
          this.konva.group.add(this.image.konva.group);
        } else if (this.image.isLoading || this.image.isError) {
          // noop
        } else {
          await this.image.update({ ...this.image.state, image }, true);
        }
        this.image.konva.group.visible(shouldShowStagedImage);
      } else {
        this.image?.destroy();
        this.image = null;
      }
    } finally {
      release();
    }
  };

  getNodes = () => {
    return [this.konva.group];
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    if (this.image) {
      this.image.destroy();
    }
    for (const node of this.getNodes()) {
      node.destroy();
    }
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      $shouldShowStagedImage: this.$shouldShowStagedImage.get(),
      $isStaging: this.$isStaging.get(),
      image: this.image?.repr() ?? null,
    };
  };
}
