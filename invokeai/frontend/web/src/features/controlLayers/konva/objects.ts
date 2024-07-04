import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { getLayerBboxId, LAYER_BBOX_NAME } from 'features/controlLayers/konva/naming';
import type { BrushLine, CanvasEntity, EraserLine, ImageObject, RectShape } from 'features/controlLayers/store/types';
import { RGBA_RED } from 'features/controlLayers/store/types';
import { t } from 'i18next';
import Konva from 'konva';
import { getImageDTO as defaultGetImageDTO } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

/**
 * Creates a bounding box rect for a layer.
 * @param entity The layer state for the layer to create the bounding box for
 * @param konvaLayer The konva layer to attach the bounding box to
 */
export const createBboxRect = (entity: CanvasEntity, konvaLayer: Konva.Layer): Konva.Rect => {
  const rect = new Konva.Rect({
    id: getLayerBboxId(entity.id),
    name: LAYER_BBOX_NAME,
    strokeWidth: 1,
    visible: false,
  });
  konvaLayer.add(rect);
  return rect;
};

export class KonvaBrushLine {
  id: string;
  konvaLineGroup: Konva.Group;
  konvaLine: Konva.Line;
  lastBrushLine: BrushLine;

  constructor(brushLine: BrushLine) {
    const { id, strokeWidth, clip, color, points } = brushLine;
    this.id = id;
    this.konvaLineGroup = new Konva.Group({
      clip,
      listening: false,
    });
    this.konvaLine = new Konva.Line({
      id,
      listening: false,
      shadowForStrokeEnabled: false,
      strokeWidth,
      tension: 0,
      lineCap: 'round',
      lineJoin: 'round',
      globalCompositeOperation: 'source-over',
      stroke: rgbaColorToString(color),
      points,
    });
    this.konvaLineGroup.add(this.konvaLine);
    this.lastBrushLine = brushLine;
  }

  update(brushLine: BrushLine, force?: boolean): boolean {
    if (this.lastBrushLine !== brushLine || force) {
      const { points, color, clip, strokeWidth } = brushLine;
      this.konvaLine.setAttrs({
        points,
        stroke: rgbaColorToString(color),
        clip,
        strokeWidth,
      });
      this.lastBrushLine = brushLine;
      return true;
    } else {
      return false;
    }
  }

  destroy() {
    this.konvaLineGroup.destroy();
  }
}

export class KonvaEraserLine {
  id: string;
  konvaLineGroup: Konva.Group;
  konvaLine: Konva.Line;
  lastEraserLine: EraserLine;

  constructor(eraserLine: EraserLine) {
    const { id, strokeWidth, clip, points } = eraserLine;
    this.id = id;
    this.konvaLineGroup = new Konva.Group({
      clip,
      listening: false,
    });
    this.konvaLine = new Konva.Line({
      id,
      listening: false,
      shadowForStrokeEnabled: false,
      strokeWidth,
      tension: 0,
      lineCap: 'round',
      lineJoin: 'round',
      globalCompositeOperation: 'destination-out',
      stroke: rgbaColorToString(RGBA_RED),
      points,
    });
    this.konvaLineGroup.add(this.konvaLine);
    this.lastEraserLine = eraserLine;
  }

  update(eraserLine: EraserLine, force?: boolean): boolean {
    if (this.lastEraserLine !== eraserLine || force) {
      const { points, clip, strokeWidth } = eraserLine;
      this.konvaLine.setAttrs({
        points,
        clip,
        strokeWidth,
      });
      this.lastEraserLine = eraserLine;
      return true;
    } else {
      return false;
    }
  }

  destroy() {
    this.konvaLineGroup.destroy();
  }
}

export class KonvaRect {
  id: string;
  konvaRect: Konva.Rect;
  lastRectShape: RectShape;

  constructor(rectShape: RectShape) {
    const { id, x, y, width, height } = rectShape;
    this.id = id;
    const konvaRect = new Konva.Rect({
      id,
      x,
      y,
      width,
      height,
      listening: false,
      fill: rgbaColorToString(rectShape.color),
    });
    this.konvaRect = konvaRect;
    this.lastRectShape = rectShape;
  }

  update(rectShape: RectShape, force?: boolean): boolean {
    if (this.lastRectShape !== rectShape || force) {
      const { x, y, width, height, color } = rectShape;
      this.konvaRect.setAttrs({
        x,
        y,
        width,
        height,
        fill: rgbaColorToString(color),
      });
      this.lastRectShape = rectShape;
      return true;
    } else {
      return false;
    }
  }

  destroy() {
    this.konvaRect.destroy();
  }
}

export class KonvaImage {
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

export class KonvaProgressImage {
  id: string;
  progressImageId: string | null;
  konvaImageGroup: Konva.Group;
  konvaImage: Konva.Image | null; // The image is loaded asynchronously, so it may not be available immediately
  isLoading: boolean;
  isError: boolean;

  constructor(arg: { id: string }) {
    const { id } = arg;
    this.konvaImageGroup = new Konva.Group({ id, listening: false });

    this.id = id;
    this.progressImageId = null;
    this.konvaImage = null;
    this.isLoading = false;
    this.isError = false;
  }

  async updateImageSource(
    progressImageId: string,
    dataURL: string,
    x: number,
    y: number,
    width: number,
    height: number
  ) {
    const imageEl = new Image();
    imageEl.onload = () => {
      if (this.konvaImage) {
        this.konvaImage.setAttrs({
          image: imageEl,
          x,
          y,
          width,
          height,
        });
      } else {
        this.konvaImage = new Konva.Image({
          id: this.id,
          listening: false,
          image: imageEl,
          x,
          y,
          width,
          height,
        });
        this.konvaImageGroup.add(this.konvaImage);
      }
    };
    imageEl.id = progressImageId;
    imageEl.src = dataURL;
  }

  destroy() {
    this.konvaImageGroup.destroy();
  }
}
