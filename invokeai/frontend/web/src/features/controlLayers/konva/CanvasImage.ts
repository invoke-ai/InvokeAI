import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasObjectRenderer } from 'features/controlLayers/konva/CanvasObjectRenderer';
import { FILTER_MAP } from 'features/controlLayers/konva/filters';
import { loadImage } from 'features/controlLayers/konva/util';
import type { CanvasImageState, GetLoggingContext } from 'features/controlLayers/store/types';
import { t } from 'i18next';
import Konva from 'konva';
import type { Logger } from 'roarr';
import { getImageDTO } from 'services/api/endpoints/images';

export class CanvasImageRenderer {
  static TYPE = 'image';
  static GROUP_NAME = `${CanvasImageRenderer.TYPE}_group`;
  static IMAGE_NAME = `${CanvasImageRenderer.TYPE}_image`;
  static PLACEHOLDER_GROUP_NAME = `${CanvasImageRenderer.TYPE}_placeholder-group`;
  static PLACEHOLDER_RECT_NAME = `${CanvasImageRenderer.TYPE}_placeholder-rect`;
  static PLACEHOLDER_TEXT_NAME = `${CanvasImageRenderer.TYPE}_placeholder-text`;

  id: string;
  parent: CanvasObjectRenderer;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  state: CanvasImageState;
  konva: {
    group: Konva.Group;
    placeholder: { group: Konva.Group; rect: Konva.Rect; text: Konva.Text };
    image: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  };
  imageName: string | null;
  isLoading: boolean;
  isError: boolean;

  constructor(state: CanvasImageState, parent: CanvasObjectRenderer) {
    const { id, width, height, x, y } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.trace({ state }, 'Creating image');

    this.konva = {
      group: new Konva.Group({ name: CanvasImageRenderer.GROUP_NAME, listening: false, x, y }),
      placeholder: {
        group: new Konva.Group({ name: CanvasImageRenderer.PLACEHOLDER_GROUP_NAME, listening: false }),
        rect: new Konva.Rect({
          name: CanvasImageRenderer.PLACEHOLDER_RECT_NAME,
          fill: 'hsl(220 12% 45% / 1)', // 'base.500'
          width,
          height,
          listening: false,
        }),
        text: new Konva.Text({
          name: CanvasImageRenderer.PLACEHOLDER_TEXT_NAME,
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
          name: CanvasImageRenderer.IMAGE_NAME,
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
  };

  update = async (state: CanvasImageState, force = false): Promise<boolean> => {
    if (force || this.state !== state) {
      this.log.trace({ state }, 'Updating image');

      const { width, height, x, y, image, filters } = state;
      if (force || (this.state.image.name !== image.name && !this.isLoading)) {
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
    }

    return false;
  };

  destroy = () => {
    this.log.trace('Destroying image');
    this.konva.group.destroy();
  };

  setVisibility = (isVisible: boolean): void => {
    this.log.trace({ isVisible }, 'Setting image visibility');
    this.konva.group.visible(isVisible);
  };

  repr = () => {
    return {
      id: this.id,
      type: CanvasImageRenderer.TYPE,
      parent: this.parent.id,
      imageName: this.imageName,
      isLoading: this.isLoading,
      isError: this.isError,
      state: deepClone(this.state),
    };
  };
}
