import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import { withResultAsync } from 'common/util/result';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import { getKonvaNodeDebugAttrs, getPrefixedId } from 'features/controlLayers/konva/util';
import type {
  CanvasImageState,
  Coordinate,
  RgbaColor,
  SAMPoint,
  SAMPointLabel,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';
import type { S } from 'services/api/types';

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
  SAM_POINT_FOREGROUND_COLOR: { r: 0, g: 200, b: 0, a: 0.7 },
  SAM_POINT_BACKGROUND_COLOR: { r: 200, g: 0, b: 0, a: 0.7 },
  SAM_POINT_NEUTRAL_COLOR: { r: 0, g: 0, b: 200, a: 0.7 },
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

  $pointType = atom<SAMPointLabel>('foreground');
  $isDraggingPoint = atom<boolean>(false);

  imageState: CanvasImageState | null = null;

  points: SAMPointState[] = [];
  maskedImage: CanvasObjectImage | null = null;

  konva: {
    group: Konva.Group;
    pointGroup: Konva.Group;
    maskGroup: Konva.Group;
    compositingRect: Konva.Rect;
  };

  KONVA_CIRCLE_NAME = `${this.type}:circle`;

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
      pointGroup: new Konva.Group({ name: `${this.type}:point_group` }),
      maskGroup: new Konva.Group({ name: `${this.type}:mask_group` }),
      compositingRect: new Konva.Rect({
        name: `${this.type}:compositingRect`,
        fill: rgbaColorToString({ r: 0, g: 200, b: 200, a: 0.5 }),
        globalCompositeOperation: 'source-atop',
        listening: false,
        strokeEnabled: false,
        perfectDrawEnabled: false,
        visible: false,
      }),
    };
    this.konva.group.add(this.konva.maskGroup);
    this.konva.group.add(this.konva.pointGroup);
    this.konva.maskGroup.add(this.konva.compositingRect);
  }

  createPoint(coord: Coordinate, label: SAMPointLabel): SAMPointState {
    const id = getPrefixedId('sam_point');
    const circle = new Konva.Circle({
      name: this.KONVA_CIRCLE_NAME,
      x: Math.round(coord.x),
      y: Math.round(coord.y),
      radius: this.config.SAM_POINT_RADIUS,
      fill: rgbaColorToString(this.getSAMPointColor(label)),
      stroke: rgbaColorToString(this.config.SAM_POINT_BORDER_COLOR),
      strokeWidth: this.config.SAM_POINT_BORDER_WIDTH,
      draggable: true,
      perfectDrawEnabled: false,
    });

    circle.on('pointerup', (e) => {
      if (this.$isDraggingPoint.get()) {
        return;
      }
      e.cancelBubble = true;
      circle.destroy();
      this.points = this.points.filter((point) => point.id !== id);
    });

    circle.on('dragstart', () => {
      this.$isDraggingPoint.set(true);
    });

    circle.on('dragend', () => {
      this.$isDraggingPoint.set(false);
    });

    circle.dragBoundFunc(({ x, y }) => ({
      x: Math.round(x),
      y: Math.round(y),
    }));

    this.konva.pointGroup.add(circle);
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

  onPointerUp = (e: KonvaEventObject<PointerEvent>) => {
    if (e.evt.button !== 0) {
      return;
    }
    if (this.$isDraggingPoint.get()) {
      return;
    }
    const cursorPos = this.manager.tool.$cursorPos.get();
    if (!cursorPos) {
      return;
    }

    this.points.push(this.createPoint(cursorPos.relative, this.$pointType.get()));
  };

  setSegmentingEventListeners = () => {
    this.manager.stage.konva.stage.on('pointerup', this.onPointerUp);
  };

  removeSegmentingEventListeners = () => {
    this.manager.stage.konva.stage.off('pointerup', this.onPointerUp);
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
    for (const point of this.points) {
      point.konva.circle.destroy();
    }
    this.points = [];
    this.parent.konva.layer.add(this.konva.group);
    this.parent.konva.layer.listening(true);

    this.setSegmentingEventListeners();
  };

  process = async () => {
    this.log.trace({ points: this.getSAMPoints() }, 'Segmenting');
    const rect = this.parent.transformer.getRelativeRect();

    const rasterizeResult = await withResultAsync(() =>
      this.parent.renderer.rasterize({ rect, attrs: { filters: [], opacity: 1 } })
    );

    if (rasterizeResult.isErr()) {
      this.log.error({ error: serializeError(rasterizeResult.error) }, 'Error rasterizing entity');
      this.$isProcessing.set(false);
      return;
    }

    this.$isProcessing.set(true);

    const controller = new AbortController();
    this.abortController = controller;

    const g = new Graph(getPrefixedId('canvas_segment_anything'));
    const segmentAnything = g.addNode({
      id: getPrefixedId('segment_anything_object_identifier'),
      type: 'segment_anything_object_identifier',
      model: 'segment-anything-huge',
      image: { image_name: rasterizeResult.value.image_name },
      object_identifiers: [
        {
          points: this.getSAMPoints().map(({ x, y, label }): S['SAMPoint'] => ({
            x,
            y,
            label: label === 'foreground' ? 1 : -1,
          })),
        },
      ],
    });
    const applyMask = g.addNode({
      id: getPrefixedId('apply_tensor_mask_to_image'),
      type: 'apply_tensor_mask_to_image',
      image: { image_name: rasterizeResult.value.image_name },
    });
    g.addEdge(segmentAnything, 'mask', applyMask, 'mask');

    const segmentResult = await withResultAsync(() =>
      this.manager.stateApi.runGraphAndReturnImageOutput({
        graph: g,
        outputNodeId: applyMask.id,
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
    if (this.maskedImage) {
      this.maskedImage.destroy();
    }
    this.maskedImage = new CanvasObjectImage(this.imageState, this);
    await this.maskedImage.update(this.imageState, true);
    this.konva.compositingRect.setAttrs({
      width: this.imageState.image.width,
      height: this.imageState.image.height,
      visible: true,
    });
    this.konva.maskGroup.add(this.maskedImage.konva.group);
    this.konva.compositingRect.moveToTop();
    this.konva.maskGroup.cache();

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
    for (const point of this.points) {
      point.konva.circle.destroy();
    }
    this.points = [];
    if (this.maskedImage) {
      this.maskedImage.destroy();
    }
    this.konva.compositingRect.visible(false);
    this.konva.maskGroup.clearCache();
    this.$pointType.set('foreground');
    this.$isSegmenting.set(false);
    this.$hasProcessed.set(false);
    this.manager.stateApi.$segmentingAdapter.set(null);
    this.konva.group.remove();
    this.removeSegmentingEventListeners();
  };

  reset = () => {
    this.log.trace('Resetting segment anything');

    for (const point of this.points) {
      point.konva.circle.destroy();
    }
    this.points = [];
    if (this.maskedImage) {
      this.maskedImage.destroy();
    }
    this.konva.compositingRect.visible(false);
    this.konva.maskGroup.clearCache();

    this.abortController?.abort();
    this.abortController = null;
    this.parent.bufferRenderer.clearBuffer();
    this.parent.transformer.updatePosition();
    this.parent.renderer.syncKonvaCache(true);
    this.parent.konva.layer.listening(false);
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
    this.konva.group.remove();
    this.removeSegmentingEventListeners();
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
      konva: {
        group: getKonvaNodeDebugAttrs(this.konva.group),
        compositingRect: getKonvaNodeDebugAttrs(this.konva.compositingRect),
      },
    };
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.removeSegmentingEventListeners();
    this.konva.group.destroy();
  };
}
