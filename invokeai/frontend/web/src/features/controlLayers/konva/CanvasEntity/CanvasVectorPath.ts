import type { Rect } from 'features/controlLayers/store/types';
import Konva from 'konva';

// Store analytic bounds in an attribute so Konva clones keep them for transform previews and exports.
export class CanvasVectorPath extends Konva.Path {
  override getSelfRect(): Rect {
    return this.getAttr('vectorBounds') ?? super.getSelfRect();
  }
}
