import { rgbaColorToString } from 'common/util/colorCodeTransformers';
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
    this.parent.log.trace(`Creating eraser line ${this.id}`);

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

  async update(state: EraserLine, force?: boolean): Promise<boolean> {
    if (force || this.state !== state) {
      this.parent.log.trace(`Updating eraser line ${this.id}`);
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
    this.parent.log.trace(`Destroying eraser line ${this.id}`);
    this.konva.group.destroy();
  }
}
