import { Mutex } from 'async-mutex';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityRenderer } from 'features/controlLayers/konva/CanvasEntityRenderer';
import type { CanvasFilterModule } from 'features/controlLayers/konva/CanvasFilterModule';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleABC } from 'features/controlLayers/konva/CanvasModuleABC';
import type { CanvasStagingAreaModule } from 'features/controlLayers/konva/CanvasStagingAreaModule';
import { loadImage } from 'features/controlLayers/konva/util';
import type { CanvasImageState } from 'features/controlLayers/store/types';
import { t } from 'i18next';
import Konva from 'konva';
import type { Logger } from 'roarr';
import { getImageDTO } from 'services/api/endpoints/images';

export class CanvasObjectImageRenderer extends CanvasModuleABC {
  readonly type = 'object_image_renderer';

  id: string;
  path: string[];
  parent: CanvasEntityRenderer | CanvasStagingAreaModule | CanvasFilterModule;
  manager: CanvasManager;
  log: Logger;
  subscriptions = new Set<() => void>();

  state: CanvasImageState;
  konva: {
    group: Konva.Group;
    placeholder: { group: Konva.Group; rect: Konva.Rect; text: Konva.Text };
    image: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  };
  imageElement: HTMLImageElement | null = null;
  isLoading: boolean = false;
  isError: boolean = false;
  mutex = new Mutex();

  constructor(state: CanvasImageState, parent: CanvasEntityRenderer | CanvasStagingAreaModule | CanvasFilterModule) {
    super();
    const { id, image } = state;
    const { width, height } = image;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.parent.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.debug({ state }, 'Creating image renderer module');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      placeholder: {
        group: new Konva.Group({ name: `${this.type}:placeholder_group`, listening: false }),
        rect: new Konva.Rect({
          name: `${this.type}:placeholder_rect`,
          fill: 'hsl(220 12% 45% / 1)', // 'base.500'
          width,
          height,
          listening: false,
        }),
        text: new Konva.Text({
          name: `${this.type}:placeholder_text`,
          fill: 'hsl(220 12% 10% / 1)', // 'base.900'
          width,
          height,
          align: 'center',
          verticalAlign: 'middle',
          fontFamily: '"Inter Variable", sans-serif',
          fontSize: width / 16,
          fontStyle: '600',
          text: t('common.loadingImage', 'Loading Image'),
          listening: false,
        }),
      },
      image: null,
    };
    this.konva.placeholder.group.add(this.konva.placeholder.rect);
    this.konva.placeholder.group.add(this.konva.placeholder.text);
    this.konva.group.add(this.konva.placeholder.group);
    this.state = state;
  }

  updateImageSource = async (imageName: string) => {
    try {
      this.log.trace({ imageName }, 'Updating image source');

      this.isLoading = true;
      this.konva.group.visible(true);

      if (!this.konva.image) {
        this.konva.placeholder.group.visible(false);
        this.konva.placeholder.text.text(t('common.loadingImage', 'Loading Image'));
      }

      const imageDTO = await getImageDTO(imageName);
      if (imageDTO === null) {
        this.onFailedToLoadImage();
        return;
      }

      this.imageElement = await loadImage(imageDTO.image_url);
      await this.updateImageElement();
    } catch {
      this.onFailedToLoadImage();
    }
  };

  onFailedToLoadImage = () => {
    this.log({ image: this.state.image }, 'Failed to load image');
    this.konva.image?.visible(false);
    this.isLoading = false;
    this.isError = true;
    this.konva.placeholder.text.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
    this.konva.placeholder.group.visible(true);
  };

  updateImageElement = async () => {
    const release = await this.mutex.acquire();

    try {
      if (this.imageElement) {
        const { width, height } = this.state.image;

        if (this.konva.image) {
          this.log.trace('Updating Konva image attrs');
          this.konva.image.setAttrs({
            image: this.imageElement,
            width,
            height,
          });
        } else {
          this.log.trace('Creating new Konva image');
          this.konva.image = new Konva.Image({
            name: `${this.type}:image`,
            listening: false,
            image: this.imageElement,
            width,
            height,
          });
          this.konva.group.add(this.konva.image);
        }

        this.konva.placeholder.rect.setAttrs({ width, height });
        this.konva.placeholder.text.setAttrs({ width, height, fontSize: width / 16 });

        this.isLoading = false;
        this.isError = false;
        this.konva.placeholder.group.visible(false);
      }
    } finally {
      release();
    }
  };

  update = async (state: CanvasImageState, force = false): Promise<boolean> => {
    if (force || this.state !== state) {
      this.log.trace({ state }, 'Updating image');

      const { image } = state;
      const { width, height, image_name } = image;
      if (force || (this.state.image.image_name !== image_name && !this.isLoading)) {
        await this.updateImageSource(image_name);
      }
      this.konva.image?.setAttrs({ width, height });
      this.state = state;
      return true;
    }

    return false;
  };

  destroy = () => {
    this.log.debug('Destroying image renderer module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.group.destroy();
  };

  setVisibility = (isVisible: boolean): void => {
    this.log.trace({ isVisible }, 'Setting image visibility');
    this.konva.group.visible(isVisible);
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      isLoading: this.isLoading,
      isError: this.isError,
      state: deepClone(this.state),
    };
  };

  getLoggingContext = () => {
    return { ...this.parent.getLoggingContext(), path: this.path.join('.') };
  };
}
