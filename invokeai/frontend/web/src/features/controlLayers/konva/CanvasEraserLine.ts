import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { EraserLine } from 'features/controlLayers/store/types';
import { RGBA_RED } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasEraserLine {
  static NAME_PREFIX = 'eraser-line';
  static GROUP_NAME = `${CanvasEraserLine.NAME_PREFIX}_group`;
  static LINE_NAME = `${CanvasEraserLine.NAME_PREFIX}_line`;

  private state: EraserLine;

  id: string;
  konva: {
    group: Konva.Group;
    line: Konva.Line;
  };

  constructor(state: EraserLine) {
    const { id, strokeWidth, clip, points } = state;
    this.id = id;
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
    if (this.state !== state || force) {
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
    this.konva.group.destroy();
  }
}
