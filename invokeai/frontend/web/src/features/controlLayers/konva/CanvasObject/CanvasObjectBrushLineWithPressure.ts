import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import {
  appendPressureStrokeRenderOpsToCanvas,
  getPressureStrokeRenderOps,
  getPressureStrokeRenderOpsFromPointIndex,
  type PressureStrokeCanvasTarget,
  renderPressureStrokeToCanvas,
} from 'features/controlLayers/konva/pressure';
import { getSVGPathDataFromPoints } from 'features/controlLayers/konva/util';
import type { CanvasBrushLineWithPressureState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { NodeConfig } from 'konva/lib/Node';
import type { Logger } from 'roarr';

type GlobalCompositeOperation = NonNullable<NodeConfig['globalCompositeOperation']>;

export class CanvasObjectBrushLineWithPressure extends CanvasModuleBase {
  readonly type = 'object_brush_line_with_pressure';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  state: CanvasBrushLineWithPressureState;
  pressurePreview: {
    target: PressureStrokeCanvasTarget | null;
    renderedPointCount: number;
    previewKey: string | null;
  };
  konva: {
    group: Konva.Group;
    line: Konva.Path;
    pressureImage: Konva.Image;
  };

  constructor(
    state: CanvasBrushLineWithPressureState,
    parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer
  ) {
    super();
    const { id, clip } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug({ state }, 'Creating module');

    this.konva = {
      group: new Konva.Group({
        name: `${this.type}:group`,
        clip,
        listening: false,
      }),
      line: new Konva.Path({
        name: `${this.type}:path`,
        listening: false,
        shadowForStrokeEnabled: false,
        globalCompositeOperation: (state.globalCompositeOperation ?? 'source-over') as GlobalCompositeOperation,
        perfectDrawEnabled: false,
      }),
      pressureImage: new Konva.Image({
        name: `${this.type}:pressure_image`,
        image: document.createElement('canvas'),
        listening: false,
        visible: false,
        globalCompositeOperation: (state.globalCompositeOperation ?? 'source-over') as GlobalCompositeOperation,
        perfectDrawEnabled: false,
      }),
    };
    this.konva.group.add(this.konva.line, this.konva.pressureImage);
    this.state = state;
    this.pressurePreview = {
      target: null,
      renderedPointCount: 0,
      previewKey: null,
    };
  }

  getPressurePreviewKey = (state: CanvasBrushLineWithPressureState): string => {
    const { id, strokeWidth, color, pressureAffectsWidth, pressureAffectsOpacity, globalCompositeOperation } = state;

    return [
      id,
      strokeWidth,
      color.r,
      color.g,
      color.b,
      color.a,
      pressureAffectsWidth,
      pressureAffectsOpacity,
      globalCompositeOperation ?? 'source-over',
    ].join(':');
  };

  resetPressurePreview = () => {
    this.pressurePreview.target = null;
    this.pressurePreview.renderedPointCount = 0;
    this.pressurePreview.previewKey = null;
  };

  syncPressureImage = (arg: { canvas: HTMLCanvasElement; x: number; y: number; globalCompositeOperation?: string }) => {
    const { canvas, x, y, globalCompositeOperation } = arg;
    this.konva.pressureImage.setAttrs({
      image: canvas,
      x,
      y,
      width: canvas.width,
      height: canvas.height,
      visible: true,
      globalCompositeOperation: (globalCompositeOperation ?? 'source-over') as GlobalCompositeOperation,
    });
    this.konva.pressureImage.getLayer()?.batchDraw();
  };

  updatePressureImage = () => {
    const { points, color, strokeWidth, globalCompositeOperation, pressureAffectsWidth, pressureAffectsOpacity } =
      this.state;
    const renderOps = getPressureStrokeRenderOps({
      points,
      strokeWidth,
      color,
      pressureAffectsWidth,
      pressureAffectsOpacity,
    });

    const rasterizedStroke = renderPressureStrokeToCanvas(renderOps);

    if (!rasterizedStroke) {
      this.resetPressurePreview();
      this.konva.pressureImage.setAttrs({
        visible: false,
        width: 0,
        height: 0,
      });
      return;
    }

    this.syncPressureImage({
      canvas: rasterizedStroke.canvas,
      x: rasterizedStroke.x,
      y: rasterizedStroke.y,
      globalCompositeOperation,
    });
  };

  updatePressureImagePreview = () => {
    const { points, color, strokeWidth, globalCompositeOperation, pressureAffectsWidth, pressureAffectsOpacity } =
      this.state;
    const pointCount = Math.floor(points.length / 3);
    const previewKey = this.getPressurePreviewKey(this.state);
    const shouldResetPreview =
      this.pressurePreview.target === null ||
      this.pressurePreview.previewKey !== previewKey ||
      pointCount <= this.pressurePreview.renderedPointCount;

    if (shouldResetPreview) {
      this.resetPressurePreview();
      const renderOps = getPressureStrokeRenderOps({
        points,
        strokeWidth,
        color,
        pressureAffectsWidth,
        pressureAffectsOpacity,
      });
      const target = appendPressureStrokeRenderOpsToCanvas(null, renderOps);

      if (!target) {
        this.konva.pressureImage.setAttrs({
          visible: false,
          width: 0,
          height: 0,
        });
        return;
      }

      this.pressurePreview.target = target;
      this.pressurePreview.renderedPointCount = pointCount;
      this.pressurePreview.previewKey = previewKey;
      this.syncPressureImage({
        canvas: target.canvas,
        x: target.x,
        y: target.y,
        globalCompositeOperation,
      });
      return;
    }

    const incrementalRenderOps = getPressureStrokeRenderOpsFromPointIndex({
      points,
      strokeWidth,
      color,
      pressureAffectsWidth,
      pressureAffectsOpacity,
      startPointIndex: Math.max(0, this.pressurePreview.renderedPointCount - 1),
    });
    const target = appendPressureStrokeRenderOpsToCanvas(this.pressurePreview.target, incrementalRenderOps);

    if (!target) {
      this.updatePressureImage();
      return;
    }

    this.pressurePreview.target = target;
    this.pressurePreview.renderedPointCount = pointCount;
    this.pressurePreview.previewKey = previewKey;
    this.syncPressureImage({
      canvas: target.canvas,
      x: target.x,
      y: target.y,
      globalCompositeOperation,
    });
  };

  finalizePressureImage = () => {
    if (!this.state.pressureAffectsOpacity) {
      return;
    }

    this.resetPressurePreview();
    this.updatePressureImage();
  };

  shouldUseNativePressurePreview = () => this.parent.type === 'buffer_renderer';

  update(state: CanvasBrushLineWithPressureState, force = false): boolean {
    if (force || this.state !== state) {
      this.log.trace({ state }, 'Updating brush line with pressure');
      const { points, color, strokeWidth, globalCompositeOperation, pressureAffectsWidth, pressureAffectsOpacity } =
        state;
      this.konva.line.visible(!pressureAffectsOpacity);
      this.konva.pressureImage.visible(pressureAffectsOpacity);
      this.konva.line.setAttrs({
        globalCompositeOperation: (globalCompositeOperation ?? 'source-over') as GlobalCompositeOperation,
        data: getSVGPathDataFromPoints(points, {
          size: strokeWidth / 2,
          simulatePressure: false,
          last: true,
          thinning: pressureAffectsWidth ? 1 : 0,
        }),
        fill: rgbaColorToString(color),
      });
      this.state = state;
      if (pressureAffectsOpacity) {
        if (this.shouldUseNativePressurePreview()) {
          this.updatePressureImagePreview();
        } else {
          this.updatePressureImage();
        }
      } else {
        this.resetPressurePreview();
        this.konva.pressureImage.setAttrs({
          visible: false,
          width: 0,
          height: 0,
        });
      }
      return true;
    }

    return false;
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting brush line visibility');
    this.konva.group.visible(isVisible);
  }

  destroy = () => {
    this.log.debug('Destroying module');
    this.konva.group.destroy();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      state: deepClone(this.state),
    };
  };
}
