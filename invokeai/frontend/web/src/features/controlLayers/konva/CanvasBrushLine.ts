import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { BrushLine } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasBrushLine {
  static NAME_PREFIX = 'brush-line';
  static GROUP_NAME = `${CanvasBrushLine.NAME_PREFIX}_group`;
  static LINE_NAME = `${CanvasBrushLine.NAME_PREFIX}_line`;

  private state: BrushLine;

  id: string;
  konva: {
    group: Konva.Group;
    line: Konva.Line;
  };

  constructor(state: BrushLine) {
    this.state = state;
    const { id, strokeWidth, clip, color, points } = this.state;
    this.id = id;
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

  async update(state: BrushLine, force?: boolean): Promise<boolean> {
    if (force || this.state !== state) {
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
    this.konva.group.destroy();
  }
}
