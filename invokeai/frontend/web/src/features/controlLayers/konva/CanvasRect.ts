import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { GetLoggingContext, RectShape } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasRect {
  static TYPE = 'rect';
  static GROUP_NAME = `${CanvasRect.TYPE}_group`;
  static RECT_NAME = `${CanvasRect.TYPE}_rect`;

  id: string;
  parent: CanvasLayer;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  state: RectShape;
  konva: {
    group: Konva.Group;
    rect: Konva.Rect;
  };

  constructor(state: RectShape, parent: CanvasLayer) {
    const { id, x, y, width, height, color } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.getLoggingContext = this.manager.buildObjectGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.trace({ state }, 'Creating rect');

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
      this.log.trace({ state }, 'Updating rect');
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
    this.log.trace('Destroying rect');
    this.konva.group.destroy();
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting rect visibility');
    this.konva.group.visible(isVisible);
  }

  repr() {
    return {
      id: this.id,
      type: CanvasRect.TYPE,
      parent: this.parent.id,
      state: deepClone(this.state),
    };
  }
}
