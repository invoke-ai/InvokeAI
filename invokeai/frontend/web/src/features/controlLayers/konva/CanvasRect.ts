import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import { CanvasObject } from 'features/controlLayers/konva/CanvasObject';
import type { RectShape } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasRect extends CanvasObject {
  static NAME_PREFIX = 'canvas-rect';
  static GROUP_NAME = `${CanvasRect.NAME_PREFIX}_group`;
  static RECT_NAME = `${CanvasRect.NAME_PREFIX}_rect`;
  static TYPE = 'rect';

  state: RectShape;
  konva: {
    group: Konva.Group;
    rect: Konva.Rect;
  };

  constructor(state: RectShape, parent: CanvasLayer) {
    super(state.id, parent);
    this._log.trace({ state }, 'Creating rect');

    const { x, y, width, height, color } = state;

    this.konva = {
      group: new Konva.Group({ name: CanvasRect.GROUP_NAME, listening: false }),
      rect: new Konva.Rect({
        name: CanvasRect.RECT_NAME,
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

  update(state: RectShape, force?: boolean): boolean {
    if (this.state !== state || force) {
      this._log.trace({ state }, 'Updating rect');
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
    this._log.trace('Destroying rect');
    this.konva.group.destroy();
  }

  setVisibility(isVisible: boolean): void {
    this._log.trace({ isVisible }, 'Setting rect visibility');
    this.konva.group.visible(isVisible);
  }

  repr() {
    return {
      id: this.id,
      type: CanvasRect.TYPE,
      parent: this._parent.id,
      state: deepClone(this.state),
    };
  }
}
