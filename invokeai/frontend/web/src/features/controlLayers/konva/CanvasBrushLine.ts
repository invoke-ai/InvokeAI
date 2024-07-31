import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { BrushLine } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasBrushLine {
  static NAME_PREFIX = 'brush-line';
  static GROUP_NAME = `${CanvasBrushLine.NAME_PREFIX}_group`;
  static LINE_NAME = `${CanvasBrushLine.NAME_PREFIX}_line`;

  state: BrushLine;

  type = 'brush_line';
  id: string;
  konva: {
    group: Konva.Group;
    line: Konva.Line;
  };

  parent: CanvasLayer;

  constructor(state: BrushLine, parent: CanvasLayer) {
    const { id, strokeWidth, clip, color, points } = state;

    this.id = id;

    this.parent = parent;
    this.parent._log.trace(`Creating brush line ${this.id}`);

    this.konva = {
      group: new Konva.Group({
        name: CanvasBrushLine.GROUP_NAME,
        clip,
        listening: false,
      }),
      line: new Konva.Line({
        name: CanvasBrushLine.LINE_NAME,
        id,
        listening: false,
        shadowForStrokeEnabled: false,
        strokeWidth,
        tension: 0,
        lineCap: 'round',
        lineJoin: 'round',
        globalCompositeOperation: 'source-over',
        stroke: rgbaColorToString(color),
        // A line with only one point will not be rendered, so we duplicate the points to make it visible
        points: points.length === 2 ? [...points, ...points] : points,
      }),
    };
    this.konva.group.add(this.konva.line);
    this.state = state;
  }

  update(state: BrushLine, force?: boolean): boolean {
    if (force || this.state !== state) {
      this.parent._log.trace(`Updating brush line ${this.id}`);
      const { points, color, clip, strokeWidth } = state;
      this.konva.line.setAttrs({
        // A line with only one point will not be rendered, so we duplicate the points to make it visible
        points: points.length === 2 ? [...points, ...points] : points,
        stroke: rgbaColorToString(color),
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
    this.parent._log.trace(`Destroying brush line ${this.id}`);
    this.konva.group.destroy();
  }

  show() {
    this.konva.group.visible(true);
  }

  hide() {
    this.konva.group.visible(false);
  }

  repr() {
    return {
      id: this.id,
      type: this.type,
      parent: this.parent.id,
      state: deepClone(this.state),
    };
  }
}
