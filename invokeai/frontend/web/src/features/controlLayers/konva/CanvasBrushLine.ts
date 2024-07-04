import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type { BrushLine } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasBrushLine {
  id: string;
  konvaLineGroup: Konva.Group;
  konvaLine: Konva.Line;
  lastBrushLine: BrushLine;

  constructor(brushLine: BrushLine) {
    const { id, strokeWidth, clip, color, points } = brushLine;
    this.id = id;
    this.konvaLineGroup = new Konva.Group({
      clip,
      listening: false,
    });
    this.konvaLine = new Konva.Line({
      id,
      listening: false,
      shadowForStrokeEnabled: false,
      strokeWidth,
      tension: 0,
      lineCap: 'round',
      lineJoin: 'round',
      globalCompositeOperation: 'source-over',
      stroke: rgbaColorToString(color),
      points,
    });
    this.konvaLineGroup.add(this.konvaLine);
    this.lastBrushLine = brushLine;
  }

  update(brushLine: BrushLine, force?: boolean): boolean {
    if (this.lastBrushLine !== brushLine || force) {
      const { points, color, clip, strokeWidth } = brushLine;
      this.konvaLine.setAttrs({
        points,
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
