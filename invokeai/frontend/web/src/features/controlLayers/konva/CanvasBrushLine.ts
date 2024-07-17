import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { BrushLine } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasBrushLine {
  static NAME_PREFIX = 'brush-line';
  static GROUP_NAME = `${CanvasBrushLine.NAME_PREFIX}_group`;
  static LINE_NAME = `${CanvasBrushLine.NAME_PREFIX}_line`;

  id: string;
  konvaLineGroup: Konva.Group;
  konvaLine: Konva.Line;
  lastBrushLine: BrushLine;

  constructor(brushLine: BrushLine) {
    const { id, strokeWidth, clip, color, points } = brushLine;
    this.id = id;
    this.konvaLineGroup = new Konva.Group({
      name: CanvasBrushLine.GROUP_NAME,
      clip,
      listening: false,
    });
    this.konvaLine = new Konva.Line({
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
    });
    this.konvaLineGroup.add(this.konvaLine);
    this.lastBrushLine = brushLine;
  }

  update(brushLine: BrushLine, force?: boolean): boolean {
    if (this.lastBrushLine !== brushLine || force) {
      const { points, color, clip, strokeWidth } = brushLine;
      this.konvaLine.setAttrs({
        // A line with only one point will not be rendered, so we duplicate the points to make it visible
        points: points.length === 2 ? [...points, ...points] : points,
        stroke: rgbaColorToString(color),
        clip,
        strokeWidth,
      });
      this.lastBrushLine = brushLine;
      return true;
    } else {
      return false;
    }
  }

  destroy() {
    this.konvaLineGroup.destroy();
  }
}
