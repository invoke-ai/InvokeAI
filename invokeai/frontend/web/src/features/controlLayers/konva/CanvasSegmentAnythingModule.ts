import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import { withResultAsync } from 'common/util/result';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import {
  addCoords,
  floorCoord,
  getKonvaNodeDebugAttrs,
  getPrefixedId,
  offsetCoord,
} from 'features/controlLayers/konva/util';
import type {
  CanvasImageState,
  Coordinate,
  RgbaColor,
  SAMPoint,
  SAMPointLabel,
} from 'features/controlLayers/store/types';
import { SAM_POINT_LABEL_NUMBER_TO_STRING } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Atom } from 'nanostores';
import { atom, computed } from 'nanostores';
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
  SAM_POINT_RADIUS: 8,
  SAM_POINT_BORDER_WIDTH: 2,
  SAM_POINT_BORDER_COLOR: { r: 0, g: 0, b: 0, a: 1 },
  SAM_POINT_FOREGROUND_COLOR: { r: 50, g: 255, b: 0, a: 1 }, // green-ish
  SAM_POINT_BACKGROUND_COLOR: { r: 255, g: 0, b: 50, a: 1 }, // red-ish
  SAM_POINT_NEUTRAL_COLOR: { r: 0, g: 225, b: 255, a: 1 }, // cyan-ish
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

  $pointType = atom<SAMPointLabel>(1);
  $pointTypeEnglish = computed<(typeof SAM_POINT_LABEL_NUMBER_TO_STRING)[SAMPointLabel], Atom<SAMPointLabel>>(
    this.$pointType,
    (pointType) => SAM_POINT_LABEL_NUMBER_TO_STRING[pointType]
  );
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
    this.subscriptions.add(
      this.manager.stage.$stageAttrs.listen((stageAttrs, oldStageAttrs) => {
        if (stageAttrs.scale !== oldStageAttrs.scale) {
          this.syncPointScales();
        }
      })
    );
  }

  syncCursorStyle = () => {
    this.manager.stage.setCursor('crosshair');
  };

  createPoint(coord: Coordinate, label: SAMPointLabel): SAMPointState {
    const id = getPrefixedId('sam_point');
    const circle = new Konva.Circle({
      name: this.KONVA_CIRCLE_NAME,
      x: Math.round(coord.x),
      y: Math.round(coord.y),
      radius: this.manager.stage.unscale(this.config.SAM_POINT_RADIUS),
      fill: rgbaColorToString(this.getSAMPointColor(label)),
      stroke: rgbaColorToString(this.config.SAM_POINT_BORDER_COLOR),
      strokeWidth: this.manager.stage.unscale(this.config.SAM_POINT_BORDER_WIDTH),
      draggable: true,
      perfectDrawEnabled: true,
      opacity: 0.6,
      dragDistance: 3,
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
      this.log.trace(
        { x: Math.round(circle.x()), y: Math.round(circle.y()), label: SAM_POINT_LABEL_NUMBER_TO_STRING[label] },
        'SAM point moved'
      );
    });

    circle.dragBoundFunc((pos) => floorCoord(pos));

    this.konva.pointGroup.add(circle);

    this.log.trace(
      { x: Math.round(circle.x()), y: Math.round(circle.y()), label: SAM_POINT_LABEL_NUMBER_TO_STRING[label] },
      'Created SAM point'
    );

    return {
      id,
      label,
      konva: { circle },
    };
  }

  syncPointScales = () => {
    const radius = this.manager.stage.unscale(this.config.SAM_POINT_RADIUS);
    const borderWidth = this.manager.stage.unscale(this.config.SAM_POINT_BORDER_WIDTH);
    for (const {
      konva: { circle },
    } of this.points) {
      circle.radius(radius);
      circle.strokeWidth(borderWidth);
    }
  };

  getSAMPoints = (): SAMPoint[] => {
    const points: SAMPoint[] = [];

    for (const { konva, label } of this.points) {
      points.push({
        x: Math.round(konva.circle.x()),
        y: Math.round(konva.circle.y()),
        label,
      });
    }

    return points;
  };

  onPointerUp = (e: KonvaEventObject<PointerEvent>) => {
    if (e.evt.button !== 0) {
      return;
    }
    if (this.manager.stage.getIsDragging()) {
      return;
    }
    if (this.$isDraggingPoint.get()) {
      return;
    }
    const cursorPos = this.manager.tool.$cursorPos.get();
    if (!cursorPos) {
      return;
    }

    // We need to offset the cursor position by the parent entity's position + pixel rect to get the correct position
    const pixelRect = this.parent.transformer.$pixelRect.get();
    const position = addCoords(this.parent.state.position, pixelRect);

    const normalizedPoint = offsetCoord(cursorPos.relative, position);
    const samPoint = this.createPoint(normalizedPoint, this.$pointType.get());
    this.points.push(samPoint);
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
    this.$pointType.set(1);
    this.$isSegmenting.set(true);
    this.manager.stateApi.$segmentingAdapter.set(this.parent);
    for (const point of this.points) {
      point.konva.circle.destroy();
    }
    this.points = [];
    // Update the konva group's position to match the parent entity
    const pixelRect = this.parent.transformer.$pixelRect.get();
    const position = addCoords(this.parent.state.position, pixelRect);
    this.konva.group.setAttrs(position);
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
      object_identifiers: [{ points: this.getSAMPoints() }],
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
    this.$hasProcessed.set(false);
    this.manager.stateApi.$segmentingAdapter.set(null);
    this.konva.group.remove();
    this.parent.konva.layer.listening(false);
    this.removeSegmentingEventListeners();
    this.$isSegmenting.set(false);
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
    this.imageState = null;
    this.$hasProcessed.set(false);
  };

  cancel = () => {
    this.log.trace('Stopping segment anything');
    this.reset();
    this.$isProcessing.set(false);
    this.$hasProcessed.set(false);
    this.manager.stateApi.$segmentingAdapter.set(null);
    this.konva.group.remove();
    this.parent.konva.layer.listening(false);
    this.removeSegmentingEventListeners();
    this.$isSegmenting.set(false);
  };

  getSAMPointColor(label: SAMPointLabel): RgbaColor {
    if (label === 0) {
      return this.config.SAM_POINT_NEUTRAL_COLOR;
    } else if (label === 1) {
      return this.config.SAM_POINT_FOREGROUND_COLOR;
    } else {
      // label === -1
      return this.config.SAM_POINT_BACKGROUND_COLOR;
    }
  }

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      points: this.points.map(({ id, konva, label }) => ({
        id,
        label,
        circle: getKonvaNodeDebugAttrs(konva.circle),
      })),
      config: deepClone(this.config),
      isSegmenting: this.$isSegmenting.get(),
      konva: {
        group: getKonvaNodeDebugAttrs(this.konva.group),
        compositingRect: getKonvaNodeDebugAttrs(this.konva.compositingRect),
        maskGroup: getKonvaNodeDebugAttrs(this.konva.maskGroup),
        pointGroup: getKonvaNodeDebugAttrs(this.konva.pointGroup),
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
