import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import { withResultAsync } from 'common/util/result';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasImageState, RgbaColor, SAMPoint, SAMPointLabel } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import Konva from 'konva';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';

type CanvasSegmentAnythingModuleConfig = {
  SAM_POINT_RADIUS: number;
  SAM_POINT_BORDER_WIDTH: number;
  SAM_POINT_BORDER_COLOR: RgbaColor;
  SAM_POINT_FOREGROUND_COLOR: RgbaColor;
  SAM_POINT_BACKGROUND_COLOR: RgbaColor;
  SAM_POINT_NEUTRAL_COLOR: RgbaColor;
  PROCESS_DEBOUNCE_MS: number;
};

const DEFAULT_CONFIG: CanvasSegmentAnythingModuleConfig = {
  SAM_POINT_RADIUS: 5,
  SAM_POINT_BORDER_WIDTH: 2,
  SAM_POINT_BORDER_COLOR: { r: 0, g: 0, b: 0, a: 1 },
  SAM_POINT_FOREGROUND_COLOR: { r: 0, g: 200, b: 0, a: 1 },
  SAM_POINT_BACKGROUND_COLOR: { r: 200, g: 0, b: 0, a: 1 },
  SAM_POINT_NEUTRAL_COLOR: { r: 0, g: 0, b: 200, a: 1 },
  PROCESS_DEBOUNCE_MS: 300,
};

type SAMPointState = {
  id: string;
  label: SAMPointLabel;
  konva: {
    circle: Konva.Circle;
  };
};

export class CanvasSegmentAnythingModule extends CanvasModuleBase {
  readonly type = 'canvas_segment_anything';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasSegmentAnythingModuleConfig = DEFAULT_CONFIG;

  subscriptions = new Set<() => void>();

  /**
   * The AbortController used to cancel the filter processing.
   */
  abortController: AbortController | null = null;

  $isSegmenting = atom<boolean>(false);
  $hasProcessed = atom<boolean>(false);
  $isProcessing = atom<boolean>(false);

  imageState: CanvasImageState | null = null;

  points: SAMPointState[] = [];

  konva: {
    group: Konva.Group;
  };

  constructor(parent: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group` }),
    };
  }

  createPoint(x: number, y: number, label: SAMPointLabel): SAMPointState {
    const id = getPrefixedId('sam_point');
    const circle = new Konva.Circle({
      name: `${this.type}:circle`,
      x,
      y,
      radius: this.config.SAM_POINT_RADIUS,
      fill: rgbaColorToString(this.getSAMPointColor(label)),
      stroke: rgbaColorToString(this.config.SAM_POINT_BORDER_COLOR),
      strokeWidth: this.config.SAM_POINT_BORDER_WIDTH,
      strokeScaleEnabled: false,
      draggable: true,
    });

    circle.on('mousedown', () => {
      this.points = this.points.filter((point) => point.id !== id);
      circle.destroy();
    });

    // circle.on('dragend', (e) => {
    //   console.log('dragend', circle.x(), circle.y(), e);
    // });

    // circle.dragBoundFunc(({ x, y }) => ({
    //   x: Math.round(x),
    //   y: Math.round(y),
    // }));

    this.konva.group.add(circle);
    const state: SAMPointState = {
      id,
      label,
      konva: { circle },
    };

    return state;
  }

  getSAMPoints = (): SAMPoint[] => {
    return this.points.map(({ konva: { circle }, label }) => ({
      x: circle.x(),
      y: circle.y(),
      label,
    }));
  };

  start = () => {
    const segmentingAdapter = this.manager.stateApi.$segmentingAdapter.get();
    if (segmentingAdapter) {
      this.log.error(`Already segmenting an entity: ${segmentingAdapter.id}`);
      return;
    }
    this.log.trace('Starting segment anything');
    this.$isSegmenting.set(true);
    this.manager.stateApi.$segmentingAdapter.set(this.parent);
  };

  process = async () => {
    this.$isProcessing.set(true);

    const controller = new AbortController();
    this.abortController = controller;

    const segmentResult = await withResultAsync(() =>
      this.manager.stateApi.runGraphAndReturnImageOutput({
        graph: new Graph(),
        outputNodeId: 'TODO',
        prepend: true,
        signal: controller.signal,
      })
    );

    if (segmentResult.isErr()) {
      this.log.error({ error: serializeError(segmentResult.error) }, 'Error segmenting');
      this.$isProcessing.set(false);
      this.abortController = null;
      return;
    }

    this.log.trace({ imageDTO: segmentResult.value }, 'Segmented');
    this.imageState = imageDTOToImageObject(segmentResult.value);
    await this.parent.bufferRenderer.setBuffer(this.imageState, true);

    this.$isProcessing.set(false);
    this.$hasProcessed.set(true);
    this.abortController = null;
  };

  apply = () => {
    const imageState = this.imageState;
    if (!imageState) {
      this.log.error('No image state to apply');
      return;
    }
    this.log.trace('Applying segment anything');
    this.parent.bufferRenderer.commitBuffer();
    const rect = this.parent.transformer.getRelativeRect();
    this.manager.stateApi.rasterizeEntity({
      entityIdentifier: this.parent.entityIdentifier,
      imageObject: imageState,
      position: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
      },
      replaceObjects: true,
    });
    this.imageState = null;
    this.$isSegmenting.set(false);
    this.$hasProcessed.set(false);
    this.manager.stateApi.$segmentingAdapter.set(null);
  };

  reset = () => {
    this.log.trace('Resetting segment anything');

    this.points = [];
    this.konva.group.destroyChildren();

    this.abortController?.abort();
    this.abortController = null;
    this.parent.bufferRenderer.clearBuffer();
    this.parent.transformer.updatePosition();
    this.parent.renderer.syncKonvaCache(true);
    this.imageState = null;
    this.$hasProcessed.set(false);
  };

  cancel = () => {
    this.log.trace('Stopping segment anything');
    this.reset();
    this.$isProcessing.set(false);
    this.$isSegmenting.set(false);
    this.$hasProcessed.set(false);
    this.manager.stateApi.$segmentingAdapter.set(null);
  };

  getSAMPointColor(label: SAMPointLabel): RgbaColor {
    if (label === 'neutral') {
      return this.config.SAM_POINT_NEUTRAL_COLOR;
    } else if (label === 'foreground') {
      return this.config.SAM_POINT_FOREGROUND_COLOR;
    } else {
      return this.config.SAM_POINT_BACKGROUND_COLOR;
    }
  }

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      points: this.getSAMPoints(),
      config: deepClone(this.config),
      isSegmenting: this.$isSegmenting.get(),
    };
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.group.destroy();
  };
}
