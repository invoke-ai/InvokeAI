import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPressureStrokeRenderOps, renderPressureStrokeToCanvas } from 'features/controlLayers/konva/pressure';
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
  }

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
      this.konva.pressureImage.setAttrs({
        visible: false,
        width: 0,
        height: 0,
      });
      return;
    }

    this.konva.pressureImage.setAttrs({
      image: rasterizedStroke.canvas,
      x: rasterizedStroke.x,
      y: rasterizedStroke.y,
      width: rasterizedStroke.canvas.width,
      height: rasterizedStroke.canvas.height,
      visible: true,
      globalCompositeOperation: (globalCompositeOperation ?? 'source-over') as GlobalCompositeOperation,
    });
  };

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
        this.updatePressureImage();
      } else {
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
