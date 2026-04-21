import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasPolygonState, RgbaColor } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

const getPreviewStrokeColor = (color: RgbaColor) => rgbaColorToString({ ...color, a: Math.max(color.a, 0.9) });

export class CanvasObjectPolygon extends CanvasModuleBase {
  readonly type = 'object_polygon';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  state: CanvasPolygonState;
  konva: {
    group: Konva.Group;
    fillPolygon: Konva.Line;
    previewStroke: Konva.Line;
  };

  constructor(state: CanvasPolygonState, parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer) {
    super();
    this.id = state.id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug({ state }, 'Creating module');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: false }),
      fillPolygon: new Konva.Line({
        name: `${this.type}:fill_polygon`,
        listening: false,
        closed: true,
        strokeEnabled: false,
        perfectDrawEnabled: false,
      }),
      previewStroke: new Konva.Line({
        name: `${this.type}:preview_stroke`,
        listening: false,
        closed: false,
        fillEnabled: false,
        lineCap: 'round',
        lineJoin: 'round',
        perfectDrawEnabled: false,
        strokeWidth: 1,
      }),
    };
    this.konva.group.add(this.konva.fillPolygon, this.konva.previewStroke);
    this.state = state;
  }

  update(state: CanvasPolygonState, force = false): boolean {
    if (force || this.state !== state) {
      this.log.trace({ state }, 'Updating polygon');
      const combinedPoints = state.previewPoint
        ? [...state.points, state.previewPoint.x, state.previewPoint.y]
        : state.points;
      const hasRenderablePolygon = combinedPoints.length >= 6;
      const isLiveBufferPreview = this.parent.type === 'buffer_renderer' && this.parent.state?.id === state.id;
      const fill =
        state.compositeOperation === 'destination-out' ? 'rgba(255,255,255,1)' : rgbaColorToString(state.color);

      this.konva.fillPolygon.setAttrs({
        points: combinedPoints,
        visible: hasRenderablePolygon,
        fill,
        globalCompositeOperation: state.compositeOperation,
      });

      this.konva.previewStroke.setAttrs({
        points: combinedPoints,
        visible: (Boolean(state.previewPoint) || isLiveBufferPreview) && combinedPoints.length >= 4,
        stroke: getPreviewStrokeColor(state.color),
        globalCompositeOperation: 'source-over',
      });

      this.state = state;
      return true;
    }

    return false;
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting polygon visibility');
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
