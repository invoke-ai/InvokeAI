import type { SerializableObject } from 'common/types';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasObjectRenderer } from 'features/controlLayers/konva/CanvasObjectRenderer';
import type { CanvasRectState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasRectRenderer {
  readonly type = 'rect_renderer';

  id: string;
  path: string[];
  parent: CanvasObjectRenderer;
  manager: CanvasManager;
  log: Logger;

  state: CanvasRectState;
  konva: {
    group: Konva.Group;
    rect: Konva.Rect;
  };
  isFirstRender: boolean = false;

  constructor(state: CanvasRectState, parent: CanvasObjectRenderer) {
    const { id } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.parent.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.trace({ state }, 'Creating rect');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      rect: new Konva.Rect({ name: `${this.type}:rect`, listening: false }),
    };
    this.konva.group.add(this.konva.rect);
    this.state = state;
  }

  update(state: CanvasRectState, force = false): boolean {
    if (force || this.state !== state) {
      this.isFirstRender = false;

      this.log.trace({ state }, 'Updating rect');
      const { rect, color } = state;
      this.konva.rect.setAttrs({
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
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
      type: this.type,
      parent: this.parent.id,
      isFirstRender: this.isFirstRender,
      state: deepClone(this.state),
    };
  }

  getLoggingContext = (): SerializableObject => {
    return { ...this.parent.getLoggingContext(), path: this.path.join('.') };
  };
}
