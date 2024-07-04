import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { getLayerBboxId, LAYER_BBOX_NAME } from 'features/controlLayers/konva/naming';
import type { CanvasEntity, EraserLine } from 'features/controlLayers/store/types';
import { RGBA_RED } from 'features/controlLayers/store/types';
import Konva from 'konva';

/**
 * Creates a bounding box rect for a layer.
 * @param entity The layer state for the layer to create the bounding box for
 * @param konvaLayer The konva layer to attach the bounding box to
 */
export const createBboxRect = (entity: CanvasEntity, konvaLayer: Konva.Layer): Konva.Rect => {
  const rect = new Konva.Rect({
    id: getLayerBboxId(entity.id),
    name: LAYER_BBOX_NAME,
    strokeWidth: 1,
    visible: false,
  });
  konvaLayer.add(rect);
  return rect;
};

export class CanvasEraserLine {
  id: string;
  konvaLineGroup: Konva.Group;
  konvaLine: Konva.Line;
  lastEraserLine: EraserLine;

  constructor(eraserLine: EraserLine) {
    const { id, strokeWidth, clip, points } = eraserLine;
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
      globalCompositeOperation: 'destination-out',
      stroke: rgbaColorToString(RGBA_RED),
      points,
    });
    this.konvaLineGroup.add(this.konvaLine);
    this.lastEraserLine = eraserLine;
  }

  update(eraserLine: EraserLine, force?: boolean): boolean {
    if (this.lastEraserLine !== eraserLine || force) {
      const { points, clip, strokeWidth } = eraserLine;
      this.konvaLine.setAttrs({
        points,
        clip,
        strokeWidth,
      });
      this.lastEraserLine = eraserLine;
      return true;
    } else {
      return false;
    }
  }

  destroy() {
    this.konvaLineGroup.destroy();
  }
}
