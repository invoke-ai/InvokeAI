import type { ImageObject } from 'features/controlLayers/store/types';
import { t } from 'i18next';
import Konva from 'konva';
import { getImageDTO as defaultGetImageDTO } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

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
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>;
  onLoading: () => void;
  onLoad: (imageName: string, imageEl: HTMLImageElement) => void;
  onError: () => void;
  lastImageObject: ImageObject;

  constructor(
    imageObject: ImageObject,
    options: {
      getImageDTO?: (imageName: string) => Promise<ImageDTO | null>;
      onLoading?: () => void;
      onLoad?: (konvaImage: Konva.Image) => void;
      onError?: () => void;
    }
  ) {
    const { getImageDTO, onLoading, onLoad, onError } = options;
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
    this.getImageDTO = getImageDTO ?? defaultGetImageDTO;
    this.onLoading = function () {
      this.isLoading = true;
      if (!this.konvaImage) {
        this.konvaPlaceholderGroup.visible(true);
        this.konvaPlaceholderText.text(t('common.loadingImage', 'Loading Image'));
      }
      this.konvaImageGroup.visible(true);
      if (onLoading) {
        onLoading();
      }
    };
    this.onLoad = function (imageName: string, imageEl: HTMLImageElement) {
      if (this.konvaImage) {
        this.konvaImage.setAttrs({
          image: imageEl,
        });
      } else {
        this.konvaImage = new Konva.Image({
          id: this.id,
          listening: false,
          image: imageEl,
          width,
          height,
        });
        this.konvaImageGroup.add(this.konvaImage);
      }
      this.imageName = imageName;
      this.isLoading = false;
      this.isError = false;
      this.konvaPlaceholderGroup.visible(false);
      this.konvaImageGroup.visible(true);

      if (onLoad) {
        onLoad(this.konvaImage);
      }
    };
    this.onError = function () {
      this.imageName = null;
      this.isLoading = false;
      this.isError = true;
      this.konvaPlaceholderGroup.visible(true);
      this.konvaPlaceholderText.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
      this.konvaImageGroup.visible(true);

      if (onError) {
        onError();
      }
    };
    this.lastImageObject = imageObject;
  }

  async updateImageSource(imageName: string) {
    try {
      this.onLoading();

      const imageDTO = await this.getImageDTO(imageName);
      if (!imageDTO) {
        this.onError();
        return;
      }
      const imageEl = new Image();
      imageEl.onload = () => {
        this.onLoad(imageName, imageEl);
      };
      imageEl.onerror = () => {
        this.onError();
      };
      imageEl.id = imageName;
      imageEl.src = imageDTO.image_url;
    } catch {
      this.onError();
    }
  }

  async update(imageObject: ImageObject, force?: boolean): Promise<boolean> {
    if (this.lastImageObject !== imageObject || force) {
      const { width, height, x, y, image } = imageObject;
      if (this.lastImageObject.image.name !== image.name || force) {
        await this.updateImageSource(image.name);
      }
      this.konvaImage?.setAttrs({ x, y, width, height });
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
