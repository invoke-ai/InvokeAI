import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getSVGPathDataFromPoints } from 'features/controlLayers/konva/util';
import type { CanvasEraserLineWithPressureState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasObjectEraserLineWithPressure extends CanvasModuleBase {
  readonly type = 'object_eraser_line_with_pressure';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  state: CanvasEraserLineWithPressureState;
  konva: {
    group: Konva.Group;
    line: Konva.Path;
  };

  constructor(
    state: CanvasEraserLineWithPressureState,
    parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer
  ) {
    super();
    const { id, clip } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug({ state }, 'Creating module');

    this.konva = {
      group: new Konva.Group({
        name: `${this.type}:group`,
        clip,
        listening: false,
      }),
      line: new Konva.Path({
        name: `${this.type}:path`,
        listening: false,
        fill: 'red', // Eraser lines use compositing, does not matter what color they have
        shadowForStrokeEnabled: false,
        globalCompositeOperation: 'destination-out',
        perfectDrawEnabled: false,
      }),
    };
    this.konva.group.add(this.konva.line);
    this.state = state;
  }

  update(state: CanvasEraserLineWithPressureState, force = false): boolean {
    if (force || this.state !== state) {
      this.log.trace({ state }, 'Updating eraser line with pressure');
      const { points, strokeWidth } = state;
      this.konva.line.setAttrs({
        data: getSVGPathDataFromPoints(points, {
          size: strokeWidth / 2,
          simulatePressure: false,
          last: true,
          thinning: 1,
        }),
      });
      this.state = state;
      return true;
    }

    return false;
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting eraser line visibility');
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
