import { deepClone } from 'common/util/deepClone';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasStagingArea } from 'features/controlLayers/konva/CanvasStagingArea';
import { FILTER_MAP } from 'features/controlLayers/konva/filters';
import { loadImage } from 'features/controlLayers/konva/util';
import type { GetLoggingContext, ImageObject } from 'features/controlLayers/store/types';
import { t } from 'i18next';
import Konva from 'konva';
import type { Logger } from 'roarr';
import { getImageDTO } from 'services/api/endpoints/images';

export class CanvasImage {
  static TYPE = 'image';
  static GROUP_NAME = `${CanvasImage.TYPE}_group`;
  static IMAGE_NAME = `${CanvasImage.TYPE}_image`;
  static PLACEHOLDER_GROUP_NAME = `${CanvasImage.TYPE}_placeholder-group`;
  static PLACEHOLDER_RECT_NAME = `${CanvasImage.TYPE}_placeholder-rect`;
  static PLACEHOLDER_TEXT_NAME = `${CanvasImage.TYPE}_placeholder-text`;

  id: string;
  parent: CanvasLayer | CanvasStagingArea;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  state: ImageObject;
  konva: {
    group: Konva.Group;
    placeholder: { group: Konva.Group; rect: Konva.Rect; text: Konva.Text };
    image: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  };
  imageName: string | null;
  isLoading: boolean;
  isError: boolean;

  constructor(state: ImageObject, parent: CanvasLayer | CanvasStagingArea) {
    const { id, width, height, x, y } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.trace({ state }, 'Creating image');

    this.konva = {
      group: new Konva.Group({ name: CanvasImage.GROUP_NAME, listening: false, x, y }),
      placeholder: {
        group: new Konva.Group({ name: CanvasImage.PLACEHOLDER_GROUP_NAME, listening: false }),
        rect: new Konva.Rect({
          name: CanvasImage.PLACEHOLDER_RECT_NAME,
          fill: 'hsl(220 12% 45% / 1)', // 'base.500'
          width,
          height,
          listening: false,
        }),
        text: new Konva.Text({
          name: CanvasImage.PLACEHOLDER_TEXT_NAME,
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

    this.imageName = null;
    this.isLoading = false;
    this.isError = false;
    this.state = state;
  }

  async updateImageSource(imageName: string) {
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
        this.log.error({ imageName }, 'Image not found');
        return;
      }
      const imageEl = await loadImage(imageDTO.image_url);

      if (this.konva.image) {
        this.konva.image.setAttrs({
          image: imageEl,
        });
      } else {
        this.konva.image = new Konva.Image({
          name: CanvasImage.IMAGE_NAME,
          listening: false,
          image: imageEl,
          width: this.state.width,
          height: this.state.height,
        });
        this.konva.group.add(this.konva.image);
      }

      if (this.state.filters.length > 0) {
        this.konva.image.cache();
        this.konva.image.filters(this.state.filters.map((f) => FILTER_MAP[f]));
      } else {
        this.konva.image.clearCache();
        this.konva.image.filters([]);
      }

      this.imageName = imageName;
      this.isLoading = false;
      this.isError = false;
      this.konva.placeholder.group.visible(false);
    } catch {
      this.log({ imageName }, 'Failed to load image');
      this.konva.image?.visible(false);
      this.imageName = null;
      this.isLoading = false;
      this.isError = true;
      this.konva.placeholder.text.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
      this.konva.placeholder.group.visible(true);
    }
  }

  async update(state: ImageObject, force?: boolean): Promise<boolean> {
    if (this.state !== state || force) {
      this.log.trace({ state }, 'Updating image');

      const { width, height, x, y, image, filters } = state;
      if (this.state.image.name !== image.name || force) {
        await this.updateImageSource(image.name);
      }
      this.konva.image?.setAttrs({ x, y, width, height });
      if (filters.length > 0) {
        this.konva.image?.cache();
        this.konva.image?.filters(filters.map((f) => FILTER_MAP[f]));
      } else {
        this.konva.image?.clearCache();
        this.konva.image?.filters([]);
      }
      this.konva.placeholder.rect.setAttrs({ width, height });
      this.konva.placeholder.text.setAttrs({ width, height, fontSize: width / 16 });
      this.state = state;
      return true;
    } else {
      return false;
    }
  }

  destroy() {
    this.log.trace('Destroying image');
    this.konva.group.destroy();
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting image visibility');
    this.konva.group.visible(isVisible);
  }

  repr() {
    return {
      id: this.id,
      type: CanvasImage.TYPE,
      parent: this.parent.id,
      imageName: this.imageName,
      isLoading: this.isLoading,
      isError: this.isError,
      state: deepClone(this.state),
    };
  }
}
