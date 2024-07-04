import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { RectShape } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasRect {
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
