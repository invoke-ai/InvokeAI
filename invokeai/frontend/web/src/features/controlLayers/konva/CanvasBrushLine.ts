import type { SerializableObject } from 'common/types';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasObjectRenderer } from 'features/controlLayers/konva/CanvasObjectRenderer';
import type { CanvasBrushLineState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasBrushLineRenderer {
  readonly type = 'brush_line_renderer';

  id: string;
  path: string[];
  parent: CanvasObjectRenderer;
  manager: CanvasManager;
  log: Logger;

  state: CanvasBrushLineState;
  konva: {
    group: Konva.Group;
    line: Konva.Line;
  };

  constructor(state: CanvasBrushLineState, parent: CanvasObjectRenderer) {
    const { id, clip } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.parent.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.trace({ state }, 'Creating brush line');

    this.konva = {
      group: new Konva.Group({
        name: `${this.type}:group`,
        clip,
        listening: false,
      }),
      line: new Konva.Line({
        name: `${this.type}:line`,
        listening: false,
        shadowForStrokeEnabled: false,
        tension: 0.3,
        lineCap: 'round',
        lineJoin: 'round',
        globalCompositeOperation: 'source-over',
      }),
    };
    this.konva.group.add(this.konva.line);
    this.state = state;
  }

  update(state: CanvasBrushLineState, force = false): boolean {
    if (force || this.state !== state) {
      this.log.trace({ state }, 'Updating brush line');
      const { points, color, strokeWidth } = state;
      this.konva.line.setAttrs({
        // A line with only one point will not be rendered, so we duplicate the points to make it visible
        points: points.length === 2 ? [...points, ...points] : points,
        stroke: rgbaColorToString(color),
        strokeWidth,
      });
      this.state = state;
      return true;
    }

    return false;
  }

  destroy() {
    this.log.trace('Destroying brush line');
    this.konva.group.destroy();
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting brush line visibility');
    this.konva.group.visible(isVisible);
  }

  repr() {
    return {
      id: this.id,
      type: this.type,
      parent: this.parent.id,
      state: deepClone(this.state),
    };
  }

  getLoggingContext = (): SerializableObject => {
    return { ...this.parent.getLoggingContext(), path: this.path.join('.') };
  };
}
