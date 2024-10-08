import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasRectState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasObjectRect extends CanvasModuleBase {
  readonly type = 'object_rect';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  state: CanvasRectState;
  konva: {
    group: Konva.Group;
    rect: Konva.Rect;
  };
  isFirstRender: boolean = false;

  constructor(state: CanvasRectState, parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer) {
    super();
    this.id = state.id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug({ state }, 'Creating module');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      rect: new Konva.Rect({ name: `${this.type}:rect`, listening: false, perfectDrawEnabled: false }),
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
    this.log.debug('Destroying module');
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
}
