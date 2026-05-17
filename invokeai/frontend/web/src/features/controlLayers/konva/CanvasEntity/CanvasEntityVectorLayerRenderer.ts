import { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import { buildBezierPathData } from 'features/controlLayers/util/bezierPath';
import Konva from 'konva';

const VECTOR_PATH_STROKE = 'rgba(90, 175, 255, 1)';
const VECTOR_PATH_STROKE_WIDTH = 1.5;

export class CanvasEntityVectorLayerRenderer extends CanvasEntityObjectRenderer {
  render = async (): Promise<boolean> => {
    if (this.parent.state.type !== 'vector_layer') {
      this.konva.objectGroup.destroyChildren();
      return false;
    }

    this.konva.objectGroup.destroyChildren();

    let didRender = false;

    for (const path of this.parent.state.paths) {
      if (path.points.length < 2) {
        continue;
      }

      const data = buildBezierPathData(path.points, path.isClosed);
      if (!data) {
        continue;
      }

      this.konva.objectGroup.add(
        new Konva.Path({
          name: `${this.type}:vector_path:${path.id}`,
          data,
          stroke: VECTOR_PATH_STROKE,
          strokeWidth: VECTOR_PATH_STROKE_WIDTH,
          fillEnabled: false,
          lineCap: 'round',
          lineJoin: 'round',
          listening: false,
          perfectDrawEnabled: false,
        })
      );
      didRender = true;
    }

    this.syncKonvaCache(didRender);
    return didRender;
  };

  needsPixelBbox = (): boolean => {
    return false;
  };

  hasObjects = (): boolean => {
    return this.parent.state.type === 'vector_layer'
      ? this.parent.state.paths.length > 0 || this.parent.bufferRenderer.hasBuffer()
      : this.parent.bufferRenderer.hasBuffer();
  };
}
