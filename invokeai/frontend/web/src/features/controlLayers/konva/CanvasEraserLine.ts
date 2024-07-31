import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { EraserLine } from 'features/controlLayers/store/types';
import { RGBA_RED } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasEraserLine {
  static NAME_PREFIX = 'eraser-line';
  static GROUP_NAME = `${CanvasEraserLine.NAME_PREFIX}_group`;
  static LINE_NAME = `${CanvasEraserLine.NAME_PREFIX}_line`;

  state: EraserLine;

  type = 'eraser_line';
  id: string;
  konva: {
    group: Konva.Group;
    line: Konva.Line;
  };

  parent: CanvasLayer;

  constructor(state: EraserLine, parent: CanvasLayer) {
    const { id, strokeWidth, clip, points } = state;

    this.id = id;

    this.parent = parent;
    this.parent._log.trace(`Creating eraser line ${this.id}`);

    this.konva = {
      group: new Konva.Group({
        name: CanvasEraserLine.GROUP_NAME,
        clip,
        listening: false,
      }),
      line: new Konva.Line({
        name: CanvasEraserLine.LINE_NAME,
        id,
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
      this.parent._log.trace(`Updating eraser line ${this.id}`);
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
    this.parent._log.trace(`Destroying eraser line ${this.id}`);
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
