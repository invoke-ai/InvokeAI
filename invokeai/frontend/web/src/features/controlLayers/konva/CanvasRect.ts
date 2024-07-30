import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { RectShape } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasRect {
  static NAME_PREFIX = 'canvas-rect';
  static GROUP_NAME = `${CanvasRect.NAME_PREFIX}_group`;
  static RECT_NAME = `${CanvasRect.NAME_PREFIX}_rect`;

  state: RectShape;

  type = 'rect';

  id: string;
  konva: {
    group: Konva.Group;
    rect: Konva.Rect;
  };

  parent: CanvasLayer;

  constructor(state: RectShape, parent: CanvasLayer) {
    const { id, x, y, width, height, color } = state;

    this.id = id;

    this.parent = parent;
    this.parent._log.trace(`Creating rect ${this.id}`);

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
        fill: rgbaColorToString(color),
      }),
    };
    this.konva.group.add(this.konva.rect);
    this.state = state;
  }

  async update(state: RectShape, force?: boolean): Promise<boolean> {
    if (this.state !== state || force) {
      this.parent._log.trace(`Updating rect ${this.id}`);
      const { x, y, width, height, color } = state;
      this.konva.rect.setAttrs({
        x,
        y,
        width,
        height,
        fill: rgbaColorToString(color),
      });
      this.state = state;
      return true;
    } else {
      return false;
    }
  }

  destroy() {
    this.parent._log.trace(`Destroying rect ${this.id}`);
    this.konva.group.destroy();
  }
}
