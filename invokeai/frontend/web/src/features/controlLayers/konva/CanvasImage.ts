import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import { FILTER_MAP } from 'features/controlLayers/konva/filters';
import { loadImage } from 'features/controlLayers/konva/util';
import type { ImageObject } from 'features/controlLayers/store/types';
import { t } from 'i18next';
import Konva from 'konva';
import { getImageDTO } from 'services/api/endpoints/images';
import { assert } from 'tsafe';

export class CanvasImage {
  static NAME_PREFIX = 'canvas-image';
  static GROUP_NAME = `${CanvasImage.NAME_PREFIX}_group`;
  static IMAGE_NAME = `${CanvasImage.NAME_PREFIX}_image`;
  static PLACEHOLDER_GROUP_NAME = `${CanvasImage.NAME_PREFIX}_placeholder-group`;
  static PLACEHOLDER_RECT_NAME = `${CanvasImage.NAME_PREFIX}_placeholder-rect`;
  static PLACEHOLDER_TEXT_NAME = `${CanvasImage.NAME_PREFIX}_placeholder-text`;

  state: ImageObject;

  type = 'image';

  id: string;
  konva: {
    group: Konva.Group;
    placeholder: { group: Konva.Group; rect: Konva.Rect; text: Konva.Text };
  };
  imageName: string | null;
  image: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  isLoading: boolean;
  isError: boolean;

  parent: CanvasLayer;

  constructor(state: ImageObject, parent: CanvasLayer) {
    const { id, width, height, x, y } = state;
    this.id = id;

    this.parent = parent;
    this.parent.log.trace(`Creating image ${this.id}`);

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
    };
    this.konva.placeholder.group.add(this.konva.placeholder.rect);
    this.konva.placeholder.group.add(this.konva.placeholder.text);
    this.konva.group.add(this.konva.placeholder.group);

    this.imageName = null;
    this.image = null;
    this.isLoading = false;
    this.isError = false;
    this.state = state;
  }

  async updateImageSource(imageName: string) {
    try {
      this.parent.log.trace(`Updating image source ${this.id}`);

      this.isLoading = true;
      this.konva.group.visible(true);

      if (!this.image) {
        this.konva.placeholder.group.visible(true);
        this.konva.placeholder.text.text(t('common.loadingImage', 'Loading Image'));
      }

      const imageDTO = await getImageDTO(imageName);
      assert(imageDTO !== null, 'imageDTO is null');
      const imageEl = await loadImage(imageDTO.image_url);

      if (this.image) {
        this.image.setAttrs({
          image: imageEl,
        });
      } else {
        this.image = new Konva.Image({
          name: CanvasImage.IMAGE_NAME,
          listening: false,
          image: imageEl,
          width: this.state.width,
          height: this.state.height,
        });
        this.konva.group.add(this.image);
      }

      if (this.state.filters.length > 0) {
        this.image.cache();
        this.image.filters(this.state.filters.map((f) => FILTER_MAP[f]));
      } else {
        this.image.clearCache();
        this.image.filters([]);
      }

      this.imageName = imageName;
      this.isLoading = false;
      this.isError = false;
      this.konva.placeholder.group.visible(false);
    } catch {
      this.image?.visible(false);
      this.imageName = null;
      this.isLoading = false;
      this.isError = true;
      this.konva.placeholder.text.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
      this.konva.placeholder.group.visible(true);
    }
  }

  async update(state: ImageObject, force?: boolean): Promise<boolean> {
    if (this.state !== state || force) {
      this.parent.log.trace(`Updating image ${this.id}`);

      const { width, height, x, y, image, filters } = state;
      if (this.state.image.name !== image.name || force) {
        await this.updateImageSource(image.name);
      }
      this.image?.setAttrs({ x, y, width, height });
      if (filters.length > 0) {
        this.image?.cache();
        this.image?.filters(filters.map((f) => FILTER_MAP[f]));
      } else {
        this.image?.clearCache();
        this.image?.filters([]);
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
    this.parent.log.trace(`Destroying image ${this.id}`);
    this.konva.group.destroy();
  }
}
