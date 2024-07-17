import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { RectShape } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasRect {
  static NAME_PREFIX = 'canvas-rect';
  static GROUP_NAME = `${CanvasRect.NAME_PREFIX}_group`;
  static RECT_NAME = `${CanvasRect.NAME_PREFIX}_rect`;

  id: string;
  konva: {
    group: Konva.Group;
    rect: Konva.Rect;
  };
  lastRectShape: RectShape;

  constructor(rectShape: RectShape) {
    const { id, x, y, width, height } = rectShape;
    this.id = id;
    this.konva = {
      group: new Konva.Group({ name: CanvasRect.GROUP_NAME, listening: false }),
      rect: new Konva.Rect({
        name: CanvasRect.RECT_NAME,
        id,
        x,
        y,
        width,
        height,
        listening: false,
        fill: rgbaColorToString(rectShape.color),
      }),
    };
    this.konva.group.add(this.konva.rect);
    this.lastRectShape = rectShape;
  }

  update(rectShape: RectShape, force?: boolean): boolean {
    if (this.lastRectShape !== rectShape || force) {
      const { x, y, width, height, color } = rectShape;
      this.konva.rect.setAttrs({
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
    this.konva.group.destroy();
  }
}
