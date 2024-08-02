import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasObjectRenderer } from 'features/controlLayers/konva/CanvasObjectRenderer';
import type { CanvasRectState, GetLoggingContext } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasRectRenderer {
  static TYPE = 'rect';
  static GROUP_NAME = `${CanvasRectRenderer.TYPE}_group`;
  static RECT_NAME = `${CanvasRectRenderer.TYPE}_rect`;

  id: string;
  parent: CanvasObjectRenderer;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  state: CanvasRectState;
  konva: {
    group: Konva.Group;
    rect: Konva.Rect;
  };
  isFirstRender: boolean = false;

  constructor(state: CanvasRectState, parent: CanvasObjectRenderer) {
    const { id, x, y, width, height, color } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.trace({ state }, 'Creating rect');

    this.konva = {
      group: new Konva.Group({ name: CanvasRectRenderer.GROUP_NAME, listening: false }),
      rect: new Konva.Rect({
        name: CanvasRectRenderer.RECT_NAME,
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

  update(state: CanvasRectState, force = this.isFirstRender): boolean {
    if (this.state !== state || force) {
      this.isFirstRender = false;

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
    }

    return false;
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
      type: CanvasRectRenderer.TYPE,
      parent: this.parent.id,
      isFirstRender: this.isFirstRender,
      state: deepClone(this.state),
    };
  }
}
