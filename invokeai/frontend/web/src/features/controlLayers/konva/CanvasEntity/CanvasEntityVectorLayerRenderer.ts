import { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import { areStageAttrsGonnaExplode } from 'features/controlLayers/konva/util';
import { buildBezierPathData } from 'features/controlLayers/util/bezierPath';
import Konva from 'konva';

const VECTOR_PATH_STROKE = 'rgba(90, 175, 255, 1)';
const VECTOR_PATH_STROKE_WIDTH_PX = 1.5;

export class CanvasEntityVectorLayerRenderer extends CanvasEntityObjectRenderer {
  constructor(...args: ConstructorParameters<typeof CanvasEntityObjectRenderer>) {
    super(...args);

    this.subscriptions.add(
      this.manager.stage.$stageAttrs.listen((stageAttrs, oldStageAttrs) => {
        if (areStageAttrsGonnaExplode(stageAttrs)) {
          return;
        }
        if (stageAttrs.scale !== oldStageAttrs.scale) {
          this.syncPathStrokeWidths();
        }
      })
    );
  }

  render = (): Promise<boolean> => {
    if (this.parent.state.type !== 'vector_layer') {
      this.konva.objectGroup.destroyChildren();
      return Promise.resolve(false);
    }

    this.konva.objectGroup.destroyChildren();

    let didRender = false;
    const strokeWidth = this.getPathStrokeWidth();

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
          strokeWidth,
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
    return Promise.resolve(didRender);
  };

  cloneObjectGroup = (arg: Parameters<CanvasEntityObjectRenderer['cloneObjectGroup']>[0] = {}): Konva.Group => {
    const { attrs, cache } = arg;
    const clone = this.konva.objectGroup.clone();
    if (attrs) {
      clone.setAttrs(attrs);
    }
    this.syncPathStrokeWidths(clone, VECTOR_PATH_STROKE_WIDTH_PX);
    if (clone.hasChildren()) {
      const { pixelRatio = 1, imageSmoothingEnabled = false } = cache ?? {};
      clone.cache({ pixelRatio, imageSmoothingEnabled });
    }
    return clone;
  };

  needsPixelBbox = (): boolean => {
    return false;
  };

  hasObjects = (): boolean => {
    return this.parent.state.type === 'vector_layer'
      ? this.parent.state.paths.length > 0 || this.parent.bufferRenderer.hasBuffer()
      : this.parent.bufferRenderer.hasBuffer();
  };

  private getPathStrokeWidth = () => {
    return this.manager.stage.unscale(VECTOR_PATH_STROKE_WIDTH_PX);
  };

  private syncPathStrokeWidths = (
    group: Konva.Group = this.konva.objectGroup,
    strokeWidth = this.getPathStrokeWidth()
  ) => {
    for (const node of group.getChildren()) {
      if (node instanceof Konva.Path) {
        node.strokeWidth(strokeWidth);
      }
    }
  };
}
