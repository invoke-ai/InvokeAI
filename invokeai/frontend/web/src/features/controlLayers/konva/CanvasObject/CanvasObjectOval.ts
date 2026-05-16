import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasOvalState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasObjectOval extends CanvasModuleBase {
  readonly type = 'object_oval';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  state: CanvasOvalState;
  konva: {
    group: Konva.Group;
    ellipse: Konva.Ellipse;
  };

  constructor(state: CanvasOvalState, parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer) {
    super();
    this.id = state.id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug({ state }, 'Creating module');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      ellipse: new Konva.Ellipse({
        name: `${this.type}:ellipse`,
        listening: false,
        radiusX: 0,
        radiusY: 0,
        perfectDrawEnabled: false,
      }),
    };
    this.konva.group.add(this.konva.ellipse);
    this.state = state;
  }

  update(state: CanvasOvalState, force = false): boolean {
    if (force || this.state !== state) {
      this.log.trace({ state }, 'Updating oval');
      const { rect, color, compositeOperation } = state;
      const fill = compositeOperation === 'destination-out' ? 'rgba(255,255,255,1)' : rgbaColorToString(color);
      this.konva.ellipse.setAttrs({
        x: rect.x + rect.width / 2,
        y: rect.y + rect.height / 2,
        radiusX: rect.width / 2,
        radiusY: rect.height / 2,
        fill,
        globalCompositeOperation: compositeOperation,
      });
      this.state = state;
      return true;
    }

    return false;
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting oval visibility');
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
      state: deepClone(this.state),
    };
  };
}
