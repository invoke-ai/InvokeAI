import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityRenderer } from 'features/controlLayers/konva/CanvasEntityRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleABC } from 'features/controlLayers/konva/CanvasModuleABC';
import type { CanvasRectState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasObjectRectRenderer extends CanvasModuleABC {
  readonly type = 'object_rect_renderer';

  id: string;
  path: string[];
  parent: CanvasEntityRenderer;
  manager: CanvasManager;
  log: Logger;
  subscriptions = new Set<() => void>();

  state: CanvasRectState;
  konva: {
    group: Konva.Group;
    rect: Konva.Rect;
  };
  isFirstRender: boolean = false;

  constructor(state: CanvasRectState, parent: CanvasEntityRenderer) {
    super();
    const { id } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.parent.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.debug({ state }, 'Creating rect renderer module');

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

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting rect visibility');
    this.konva.group.visible(isVisible);
  }

  destroy = () => {
    this.log.debug('Destroying rect renderer module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.group.destroy();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      isFirstRender: this.isFirstRender,
      state: deepClone(this.state),
    };
  };

  getLoggingContext = () => {
    return { ...this.parent.getLoggingContext(), path: this.path.join('.') };
  };
}
