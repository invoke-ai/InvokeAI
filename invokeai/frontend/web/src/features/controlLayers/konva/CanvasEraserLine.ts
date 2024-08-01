import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { EraserLine, GetLoggingContext } from 'features/controlLayers/store/types';
import { RGBA_RED } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasEraserLine {
  static TYPE = 'eraser_line';
  static GROUP_NAME = `${CanvasEraserLine.TYPE}_group`;
  static LINE_NAME = `${CanvasEraserLine.TYPE}_line`;

  id: string;
  parent: CanvasLayer;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  state: EraserLine;
  konva: {
    group: Konva.Group;
    line: Konva.Line;
  };

  constructor(state: EraserLine, parent: CanvasLayer) {
    const { id, strokeWidth, clip, points } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.getLoggingContext = this.manager.buildObjectGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.trace({ state }, 'Creating eraser line');

    this.konva = {
      group: new Konva.Group({
        name: CanvasEraserLine.GROUP_NAME,
        clip,
        listening: false,
      }),
      line: new Konva.Line({
        name: CanvasEraserLine.LINE_NAME,
        listening: false,
        shadowForStrokeEnabled: false,
        strokeWidth,
        tension: 0,
        lineCap: 'round',
        lineJoin: 'round',
        globalCompositeOperation: 'destination-out',
        stroke: rgbaColorToString(RGBA_RED),
        // A line with only one point will not be rendered, so we duplicate the points to make it visible
        points: points.length === 2 ? [...points, ...points] : points,
      }),
    };
    this.konva.group.add(this.konva.line);
    this.state = state;
  }

  update(state: EraserLine, force?: boolean): boolean {
    if (force || this.state !== state) {
      this.log.trace({ state }, 'Updating eraser line');
      const { points, clip, strokeWidth } = state;
      this.konva.line.setAttrs({
        // A line with only one point will not be rendered, so we duplicate the points to make it visible
        points: points.length === 2 ? [...points, ...points] : points,
        clip,
        strokeWidth,
      });
      this.state = state;
      return true;
    } else {
      return false;
    }
  }

  destroy() {
    this.log.trace('Destroying eraser line');
    this.konva.group.destroy();
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting brush line visibility');
    this.konva.group.visible(isVisible);
  }

  repr() {
    return {
      id: this.id,
      type: CanvasEraserLine.TYPE,
      parent: this.parent.id,
      state: deepClone(this.state),
    };
  }
}
