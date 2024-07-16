import { FILTER_MAP } from 'features/controlLayers/konva/filters';
import { loadImage } from 'features/controlLayers/konva/util';
import type { ImageObject } from 'features/controlLayers/store/types';
import { t } from 'i18next';
import Konva from 'konva';
import { getImageDTO } from 'services/api/endpoints/images';
import { assert } from 'tsafe';

export class CanvasImage {
  id: string;
  konvaImageGroup: Konva.Group;
  konvaPlaceholderGroup: Konva.Group;
  konvaPlaceholderRect: Konva.Rect;
  konvaPlaceholderText: Konva.Text;
  imageName: string | null;
  konvaImage: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  isLoading: boolean;
  isError: boolean;
  lastImageObject: ImageObject;

  constructor(imageObject: ImageObject) {
    const { id, width, height, x, y } = imageObject;
    this.konvaImageGroup = new Konva.Group({ id, listening: false, x, y });
    this.konvaPlaceholderGroup = new Konva.Group({ listening: false });
    this.konvaPlaceholderRect = new Konva.Rect({
      fill: 'hsl(220 12% 45% / 1)', // 'base.500'
      width,
      height,
      listening: false,
    });
    this.konvaPlaceholderText = new Konva.Text({
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
    });

    this.konvaPlaceholderGroup.add(this.konvaPlaceholderRect);
    this.konvaPlaceholderGroup.add(this.konvaPlaceholderText);
    this.konvaImageGroup.add(this.konvaPlaceholderGroup);

    this.id = id;
    this.imageName = null;
    this.konvaImage = null;
    this.isLoading = false;
    this.isError = false;
    this.lastImageObject = imageObject;
  }

  async updateImageSource(imageName: string) {
    try {
      this.isLoading = true;
      this.konvaImageGroup.visible(true);

      if (!this.konvaImage) {
        this.konvaPlaceholderGroup.visible(true);
        this.konvaPlaceholderText.text(t('common.loadingImage', 'Loading Image'));
      }

      const imageDTO = await getImageDTO(imageName);
      assert(imageDTO !== null, 'imageDTO is null');
      const imageEl = await loadImage(imageDTO.image_url);

      if (this.konvaImage) {
        this.konvaImage.setAttrs({
          image: imageEl,
        });
      } else {
        this.konvaImage = new Konva.Image({
          id: this.id,
          listening: false,
          image: imageEl,
          width: this.lastImageObject.width,
          height: this.lastImageObject.height,
        });
        this.konvaImageGroup.add(this.konvaImage);
      }

      if (this.lastImageObject.filters.length > 0) {
        this.konvaImage.cache();
        this.konvaImage.filters(this.lastImageObject.filters.map((f) => FILTER_MAP[f]));
      } else {
        this.konvaImage.clearCache();
        this.konvaImage.filters([]);
      }

      this.imageName = imageName;
      this.isLoading = false;
      this.isError = false;
      this.konvaPlaceholderGroup.visible(false);
    } catch {
      this.konvaImage?.visible(false);
      this.imageName = null;
      this.isLoading = false;
      this.isError = true;
      this.konvaPlaceholderText.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
      this.konvaPlaceholderGroup.visible(true);
    }
  }

  async update(imageObject: ImageObject, force?: boolean): Promise<boolean> {
    if (this.lastImageObject !== imageObject || force) {
      const { width, height, x, y, image, filters } = imageObject;
      if (this.lastImageObject.image.name !== image.name || force) {
        await this.updateImageSource(image.name);
      }
      this.konvaImage?.setAttrs({ x, y, width, height });
      if (filters.length > 0) {
        this.konvaImage?.cache();
        this.konvaImage?.filters(filters.map((f) => FILTER_MAP[f]));
      } else {
        this.konvaImage?.clearCache();
        this.konvaImage?.filters([]);
      }
      this.konvaPlaceholderRect.setAttrs({ width, height });
      this.konvaPlaceholderText.setAttrs({ width, height, fontSize: width / 16 });
      this.lastImageObject = imageObject;
      return true;
    } else {
      return false;
    }
  }

  destroy() {
    this.konvaImageGroup.destroy();
  }
}
