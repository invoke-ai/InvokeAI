import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import { withResultAsync } from 'common/util/result';
import { debounce } from 'es-toolkit/compat';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import {
  addCoords,
  areStageAttrsGonnaExplode,
  getKonvaNodeDebugAttrs,
  getPrefixedId,
  offsetCoord,
  roundCoord,
} from 'features/controlLayers/konva/util';
import { selectAutoProcess } from 'features/controlLayers/store/canvasSettingsSlice';
import type {
  CanvasEntityType,
  CanvasImageState,
  Coordinate,
  RgbaColor,
  SAMModel,
  SAMPointLabel,
  SAMPointLabelString,
  SAMPointWithId,
} from 'features/controlLayers/store/types';
import { SAM_POINT_LABEL_NUMBER_TO_STRING } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { toast } from 'features/toast/toast';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Rect } from 'konva/lib/shapes/Rect';
import type { Atom } from 'nanostores';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';
import type { ImageDTO } from 'services/api/types';
import stableHash from 'stable-hash';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import type { JsonObject } from 'type-fest';

type CanvasSegmentAnythingModuleConfig = {
  /**
   * The radius of the SAM point Konva circle node.
   */
  SAM_POINT_RADIUS: number;
  /**
   * The border width of the SAM point Konva circle node.
   */
  SAM_POINT_BORDER_WIDTH: number;
  /**
   * The border color of the SAM point Konva circle node.
   */
  SAM_POINT_BORDER_COLOR: RgbaColor;
  /**
   * The color of the SAM point Konva circle node when the label is 1.
   */
  SAM_POINT_FOREGROUND_COLOR: RgbaColor;
  /**
   * The color of the SAM point Konva circle node when the label is -1.
   */
  SAM_POINT_BACKGROUND_COLOR: RgbaColor;
  /**
   * The color of the SAM point Konva circle node when the label is 0.
   */
  SAM_POINT_NEUTRAL_COLOR: RgbaColor;
  /**
   * The color to use for the mask preview overlay.
   */
  MASK_COLOR: RgbaColor;
  /**
   * The debounce time in milliseconds for processing the points.
   */
  PROCESS_DEBOUNCE_MS: number;
};

const DEFAULT_CONFIG: CanvasSegmentAnythingModuleConfig = {
  SAM_POINT_RADIUS: 8,
  SAM_POINT_BORDER_WIDTH: 2,
  SAM_POINT_BORDER_COLOR: { r: 0, g: 0, b: 0, a: 1 },
  SAM_POINT_FOREGROUND_COLOR: { r: 50, g: 255, b: 0, a: 1 }, // light green
  SAM_POINT_BACKGROUND_COLOR: { r: 255, g: 0, b: 50, a: 1 }, // red-ish
  SAM_POINT_NEUTRAL_COLOR: { r: 0, g: 225, b: 255, a: 1 }, // cyan
  MASK_COLOR: { r: 0, g: 225, b: 255, a: 1 }, // cyan
  PROCESS_DEBOUNCE_MS: 1000,
};

/**
 * The state of a SAM point.
 * @property id - The unique identifier of the point.
 * @property label - The label of the point. -1 is background, 0 is neutral, 1 is foreground.
 * @property konva - The Konva node state of the point.
 * @property konva.circle - The Konva circle node of the point. The x and y coordinates for the point are derived from
 * this node.
 */
type SAMPointState = {
  id: string;
  label: SAMPointLabel;
  coord: Coordinate;
  konva: {
    circle: Konva.Circle;
  };
};

type PromptInputData = {
  type: 'prompt';
  prompt: string;
};

type VisualInputData = {
  type: 'visual';
  points: SAMPointState[];
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  } | null;
};

const hasInputData = (data: PromptInputData | VisualInputData): boolean => {
  if (data.type === 'prompt') {
    return data.prompt.trim() !== '';
  } else {
    // Visual mode has input if there are points OR a bbox
    return data.points.length > 0 || data.bbox !== null;
  }
};

/**
 * Gets the SAM points in the format expected by the segment-anything API. The x and y values are rounded to integers.
 */
const getSAMPoints = (data: VisualInputData): SAMPointWithId[] => {
  const points: SAMPointWithId[] = [];

  for (const { id, coord, label } of data.points) {
    points.push({
      id,
      x: coord.x,
      y: coord.y,
      label,
    });
  }

  return points;
};

const getHashableInputData = (data: PromptInputData | VisualInputData): JsonObject => {
  if (data.type === 'prompt') {
    return { type: 'prompt', prompt: data.prompt } as const;
  } else {
    return { type: 'visual', points: getSAMPoints(data), bbox: data.bbox } as const;
  }
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
   * The AbortController used to cancel the segment processing.
   */
  abortController: AbortController | null = null;

  /**
   * Whether the module is currently segmenting an entity.
   */
  $isSegmenting = atom<boolean>(false);

  /**
   * The hash of the last processed points. This is used to prevent re-processing the same points.
   */
  $lastProcessedHash = atom<string>('');

  /**
   * Whether the module is currently processing the points.
   */
  $isProcessing = atom<boolean>(false);

  /**
   * The type of point to create when segmenting. This is a number representation of the SAMPointLabel enum.
   */
  $pointType = atom<SAMPointLabel>(1);

  /**
   * The type of point to create when segmenting. This is a number representation of the SAMPointLabel enum.
   */
  $model = atom<SAMModel>('SAM2');

  /**
   * The type of point to create when segmenting, as a string. This is a computed value based on $pointType.
   */
  $pointTypeString = computed<SAMPointLabelString, Atom<SAMPointLabel>>(
    this.$pointType,
    (pointType) => SAM_POINT_LABEL_NUMBER_TO_STRING[pointType]
  );

  /**
   * Whether a point is currently being dragged. This is used to prevent the point additions and deletions during
   * dragging.
   */
  $isDraggingPoint = atom<boolean>(false);

  /**
   * The ephemeral image state of the processed image. Only used while segmenting.
   */
  $imageState = atom<CanvasImageState | null>(null);

  /**
   * Whether the module has an image state. This is a computed value based on $imageState.
   */
  $hasImageState = computed(this.$imageState, (imageState) => imageState !== null);

  /**
   * The input data for the module. This includes the points and bounding box for visual mode, or the prompt for
   * prompt mode.
   */
  $inputData = atom<PromptInputData | VisualInputData>({ type: 'visual', points: [], bbox: null });

  /**
   * Whether the module has points. This is a computed value based on $points.
   */
  $hasInputData = computed(this.$inputData, hasInputData);

  /**
   * Whether the module should invert the mask image.
   */
  $invert = atom<boolean>(false);

  /**
   * State for bounding box drawing (i.e. the initial drag to create a bbox - not resizing or moving an existing one)
   */
  $isBboxDrawing = atom<boolean>(false);

  /**
   * The coordinate where bbox drawing started, or null if not drawing.
   */
  $bboxStartCoord = atom<Coordinate | null>(null);

  /**
   * The coordinate where bbox dragging started, or null if not dragging.
   */
  $bboxDragStart = atom<{ x: number; y: number } | null>(null);

  /**
   * State for bbox dragging (i.e. moving an existing bbox, not drawing a new one or resizing)
   */
  $isBboxDragging = atom<boolean>(false);

  /**
   * The masked image object module, if it exists.
   */
  imageModule: CanvasObjectImage | null = null;

  /**
   * The Konva nodes for the module.
   */
  konva: {
    /**
     * The main Konva group node for the module. This is added to the parent layer on start, and removed on teardown.
     */
    group: Konva.Group;
    /**
     * The Konva group node for the SAM points.
     *
     * This is a child of the main group node, rendered above the mask group.
     */
    pointGroup: Konva.Group;
    /**
     * The Konva group node for the bounding box.
     *
     * This is a child of the main group node, rendered above the mask group.
     */
    bboxGroup: Konva.Group;
    /**
     * The Konva rect node for the bounding box.
     */
    bboxRect: Konva.Rect;
    /**
     * The Konva transformer for the bounding box.
     */
    bboxTransformer: Konva.Transformer;
    /**
     * The Konva group node for the mask image and compositing rect.
     *
     * This is a child of the main group node, rendered below the point group.
     */
    maskGroup: Konva.Group;
    /**
     * The Konva rect node for compositing the mask image.
     *
     * It's rendered with a globalCompositeOperation of 'source-atop' to preview the mask as a semi-transparent overlay.
     */
    compositingRect: Konva.Rect;
    /**
     * A tween for pulsing the mask group's opacity.
     */
    maskTween: Konva.Tween | null;
  };

  KONVA_CIRCLE_NAME = `${this.type}:circle`;
  KONVA_GROUP_NAME = `${this.type}:group`;
  KONVA_POINT_GROUP_NAME = `${this.type}:point_group`;
  KONVA_BBOX_GROUP_NAME = `${this.type}:bbox_group`;
  KONVA_BBOX_RECT_NAME = `${this.type}:bbox_rect`;
  KONVA_BBOX_TRANSFORMER_NAME = `${this.type}:bbox_transformer`;
  KONVA_MASK_GROUP_NAME = `${this.type}:mask_group`;
  KONVA_COMPOSITING_RECT_NAME = `${this.type}:compositing_rect`;

  constructor(parent: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    // Create all konva nodes
    this.konva = {
      group: new Konva.Group({ name: this.KONVA_GROUP_NAME }),
      pointGroup: new Konva.Group({ name: this.KONVA_POINT_GROUP_NAME }),
      bboxGroup: new Konva.Group({ name: this.KONVA_BBOX_GROUP_NAME, listening: true }),
      bboxRect: new Konva.Rect({
        name: this.KONVA_BBOX_RECT_NAME,
        borderDash: [5, 5],
        stroke: rgbaColorToString(this.config.MASK_COLOR),
        strokeWidth: 2,
        strokeScaleEnabled: false,
        fill: rgbaColorToString({ ...this.config.MASK_COLOR, a: 0.1 }),
        draggable: false, // Start with draggable disabled, we'll handle drag manually
        listening: true,
        visible: false,
      }),
      bboxTransformer: new Konva.Transformer({
        name: this.KONVA_BBOX_TRANSFORMER_NAME,
        borderDash: [5, 5],
        borderStroke: rgbaColorToString(this.config.MASK_COLOR),
        borderEnabled: true,
        borderStrokeWidth: 1,
        rotateEnabled: false,
        keepRatio: false,
        ignoreStroke: true,
        flipEnabled: false,
        anchorFill: rgbaColorToString(this.config.MASK_COLOR),
        anchorStroke: 'rgb(42,42,42)',
        anchorSize: 12,
        anchorCornerRadius: 3,
        listening: true,
        visible: false,
      }),
      maskGroup: new Konva.Group({ name: this.KONVA_MASK_GROUP_NAME, opacity: 0.6 }),
      compositingRect: new Konva.Rect({
        name: this.KONVA_COMPOSITING_RECT_NAME,
        fill: rgbaColorToString(this.config.MASK_COLOR),
        globalCompositeOperation: 'source-atop',
        listening: false,
        strokeEnabled: false,
        perfectDrawEnabled: false,
        visible: false,
      }),
      maskTween: null,
    };

    // Points and bbox should always be rendered above the mask group
    this.konva.group.add(this.konva.maskGroup);
    this.konva.group.add(this.konva.bboxGroup);
    this.konva.group.add(this.konva.pointGroup);

    // Add bbox rect and transformer to bbox group
    this.konva.bboxGroup.add(this.konva.bboxRect);
    this.konva.bboxGroup.add(this.konva.bboxTransformer);

    // Set the transformer to transform the bbox rect
    this.konva.bboxTransformer.nodes([this.konva.bboxRect]);

    // Increase the hit area for the bbox transformer anchors to make them easier to grab
    this.konva.bboxTransformer.find<Rect>('._anchor').forEach((node) => {
      node.hitStrokeWidth(12);
    });

    // Add event handlers for bbox transformer
    this.konva.bboxTransformer.on('transformend', () => {
      const data = this.$inputData.get();
      if (data.type !== 'visual') {
        return;
      }

      const x = this.konva.bboxRect.x();
      const y = this.konva.bboxRect.y();
      const scaleX = this.konva.bboxRect.scaleX();
      const scaleY = this.konva.bboxRect.scaleY();

      // Apply scale to dimensions, ensuring minimum size to prevent issues
      const width = Math.max(1, this.konva.bboxRect.width() * scaleX);
      const height = Math.max(1, this.konva.bboxRect.height() * scaleY);

      // Reset scale after transform
      this.konva.bboxRect.setAttrs({
        x: x,
        y: y,
        width: width,
        height: height,
        scaleX: 1,
        scaleY: 1,
      });

      this.$inputData.set({
        ...data,
        bbox: {
          x: x,
          y: y,
          width: width,
          height: height,
        },
      });
    });

    // Handle bbox dragging
    this.konva.bboxRect.on('dragend', () => {
      const data = this.$inputData.get();
      if (data.type !== 'visual') {
        return;
      }

      const x = this.konva.bboxRect.x();
      const y = this.konva.bboxRect.y();
      // When dragging (not transforming), scale should be 1, but let's be safe
      const width = this.konva.bboxRect.width() * this.konva.bboxRect.scaleX();
      const height = this.konva.bboxRect.height() * this.konva.bboxRect.scaleY();

      this.$inputData.set({
        ...data,
        bbox: {
          x: x,
          y: y,
          width: width,
          height: height,
        },
      });
    });

    // Handle manual drag detection for bbox rect
    this.konva.bboxRect.on('pointerdown', (e) => {
      // Only handle left mouse button - other buttons are for panning or context menu
      if (e.evt.button !== 0) {
        return;
      }

      const data = this.$inputData.get();
      if (data.type !== 'visual') {
        return;
      }

      // Get the position of the mouse/touch relative to the stage
      const stage = this.manager.stage.konva.stage;
      const pointerPos = stage.getPointerPosition();
      if (!pointerPos) {
        return;
      }

      // Store the initial position for drag detection
      this.$bboxDragStart.set({ x: pointerPos.x, y: pointerPos.y });
      this.$isBboxDragging.set(false);

      // Don't bubble the event yet - we'll decide what to do on move/up
      e.cancelBubble = true;
    });

    // Handle transformer interactions
    this.konva.bboxTransformer.on('pointerdown', (e) => {
      // Only handle left mouse button - other buttons are for panning or context menu
      if (e.evt.button !== 0) {
        return;
      }

      // Transformer handles its own dragging, just stop propagation
      e.cancelBubble = true;
    });

    // Compositing rect is added to the mask group - will also be above the mask image, but that doesn't get created
    // until after processing
    this.konva.maskGroup.add(this.konva.compositingRect);
  }

  /**
   * Synchronizes the cursor style to crosshair.
   */
  syncCursorStyle = (): void => {
    if (this.$isProcessing.get()) {
      this.manager.stage.setCursor('wait');
    } else if (this.$isSegmenting.get()) {
      this.manager.stage.setCursor('crosshair');
    }
  };

  /**
   * Creates a SAM point at the given coordinate with the given label. -1 is background, 0 is neutral, 1 is foreground.
   * @param coord The coordinate
   * @param label The label.
   * @returns The SAM point state.
   */
  createPoint(coord: Coordinate, label: SAMPointLabel): SAMPointState {
    const id = getPrefixedId('sam_point');

    const roundedCoord = roundCoord(coord);

    const circle = new Konva.Circle({
      name: this.KONVA_CIRCLE_NAME,
      x: roundedCoord.x,
      y: roundedCoord.y,
      radius: this.manager.stage.unscale(this.config.SAM_POINT_RADIUS), // We will scale this as the stage scale changes
      fill: rgbaColorToString(this.getSAMPointColor(label)),
      stroke: rgbaColorToString(this.config.SAM_POINT_BORDER_COLOR),
      strokeWidth: this.manager.stage.unscale(this.config.SAM_POINT_BORDER_WIDTH), // We will scale this as the stage scale changes
      draggable: true,
      perfectDrawEnabled: true, // Required for the stroke/fill to draw correctly w/ partial opacity
      opacity: 0.6,
      dragDistance: 3,
    });

    // When the point is clicked, remove it
    circle.on('pointerup', (e) => {
      // Ignore if we are dragging
      if (this.$isDraggingPoint.get()) {
        return;
      }
      if (e.evt.button !== 0) {
        return;
      }
      // This event should not bubble up to the parent, stage or any other nodes
      e.cancelBubble = true;
      circle.destroy();

      const data = this.$inputData.get();
      if (data.type !== 'visual') {
        return;
      }

      const newPoints = data.points.filter((point) => point.id !== id);
      this.$inputData.set({ ...data, points: newPoints });
    });

    circle.on('dragstart', () => {
      this.$isDraggingPoint.set(true);
    });

    circle.on('dragend', () => {
      const roundedCoord = roundCoord(circle.position());

      this.log.trace({ ...roundedCoord, label: SAM_POINT_LABEL_NUMBER_TO_STRING[label] }, 'Moved SAM point');
      this.$isDraggingPoint.set(false);

      const data = this.$inputData.get();
      if (data.type !== 'visual') {
        return;
      }

      const newPoints = data.points.map((point) => {
        if (point.id === id) {
          return { ...point, coord: roundedCoord };
        }
        return point;
      });

      this.$inputData.set({ ...data, points: newPoints });
    });

    this.konva.pointGroup.add(circle);

    this.log.trace({ ...roundedCoord, label: SAM_POINT_LABEL_NUMBER_TO_STRING[label] }, 'Created SAM point');

    return {
      id,
      coord: roundedCoord,
      label,
      konva: { circle },
    };
  }

  /**
   * Synchronizes the scales of the SAM points to the stage scale.
   *
   * SAM points are always the same size, regardless of the stage scale.
   */
  syncPointScales = () => {
    const data = this.$inputData.get();
    if (data.type !== 'visual') {
      return;
    }
    const radius = this.manager.stage.unscale(this.config.SAM_POINT_RADIUS);
    const borderWidth = this.manager.stage.unscale(this.config.SAM_POINT_BORDER_WIDTH);
    for (const point of data.points) {
      point.konva.circle.radius(radius);
      point.konva.circle.strokeWidth(borderWidth);
    }
  };

  /**
   * Synchronizes the bbox visibility based on the current input data.
   */
  syncBboxVisibility = () => {
    const data = this.$inputData.get();
    if (data.type !== 'visual') {
      return;
    }

    if (data.bbox) {
      // Update bbox position and size
      this.konva.bboxRect.setAttrs({
        x: data.bbox.x,
        y: data.bbox.y,
        width: data.bbox.width,
        height: data.bbox.height,
        visible: true,
        listening: true, // Ensure existing bboxes are interactive
      });
      this.konva.bboxTransformer.visible(true);
    } else {
      // Hide bbox if there's no bbox data
      this.konva.bboxRect.visible(false);
      this.konva.bboxTransformer.visible(false);
    }
  };

  /**
   * Handles the pointerdown event on the stage. This is used to start drawing a bounding box in visual mode.
   * We'll start tracking the position but won't decide if it's a bbox or point until pointerup.
   */
  onStagePointerDown = (e: KonvaEventObject<PointerEvent>) => {
    const data = this.$inputData.get();
    if (data.type !== 'visual') {
      return;
    }

    // Only handle left mouse button - other buttons are for panning or context menu
    if (e.evt.button !== 0) {
      return;
    }

    // Ignore if clicking on the bbox rect or transformer (let those handle their own events)
    if (e.target !== this.manager.stage.konva.stage) {
      return;
    }

    // Ignore if the stage is dragging/panning
    if (this.manager.stage.getIsDragging()) {
      return;
    }

    // Ignore if we are already processing
    if (this.$isProcessing.get()) {
      return;
    }

    // Ignore if the cursor is not within the stage
    const cursorPos = this.manager.tool.$cursorPos.get();
    if (!cursorPos) {
      return;
    }

    // We need to offset the cursor position by the parent entity's position + pixel rect to get the correct position
    const pixelRect = this.parent.transformer.$pixelRect.get();
    const parentPosition = addCoords(this.parent.state.position, pixelRect);

    // Normalize the cursor position to the parent entity's position
    const normalizedPoint = offsetCoord(cursorPos.relative, parentPosition);

    // Start potential bbox drawing (we'll decide in pointerup if it's actually a bbox or a point)
    this.$isBboxDrawing.set(true);
    this.$bboxStartCoord.set(normalizedPoint);

    // Prepare for potential new bbox but don't hide existing one yet
    // We'll only update visibility during drag if it's actually a new bbox
  };

  /**
   * Handles the pointermove event on the stage. This is used to update the bounding box while drawing.
   */
  onStagePointerMove = () => {
    const data = this.$inputData.get();
    if (data.type !== 'visual') {
      return;
    }

    if (!this.$isBboxDrawing.get()) {
      return;
    }

    const startCoord = this.$bboxStartCoord.get();
    if (!startCoord) {
      return;
    }

    // Get current cursor position
    const cursorPos = this.manager.tool.$cursorPos.get();
    if (!cursorPos) {
      return;
    }

    // We need to offset the cursor position by the parent entity's position + pixel rect to get the correct position
    const pixelRect = this.parent.transformer.$pixelRect.get();
    const parentPosition = addCoords(this.parent.state.position, pixelRect);

    // Normalize the cursor position to the parent entity's position
    const currentPoint = offsetCoord(cursorPos.relative, parentPosition);

    // Calculate the bbox dimensions
    const x = Math.min(startCoord.x, currentPoint.x);
    const y = Math.min(startCoord.y, currentPoint.y);
    const width = Math.abs(currentPoint.x - startCoord.x);
    const height = Math.abs(currentPoint.y - startCoord.y);

    // Only show the bbox and hide transformer if we've dragged more than a threshold (5 pixels)
    if (width > 5 || height > 5) {
      // Now we know it's a drag for a new bbox, hide the transformer
      this.konva.bboxTransformer.visible(false);

      // Update and show the new bbox rect
      this.konva.bboxRect.setAttrs({
        x,
        y,
        width,
        height,
        visible: true,
        listening: false, // Disable listening during drawing to prevent event interception
      });
    }
  };

  /**
   * Handles the pointerup event on the stage. This is used to add a SAM point or finish drawing a bounding box.
   */
  onStagePointerUp = (e: KonvaEventObject<PointerEvent>) => {
    const data = this.$inputData.get();

    // Handle visual mode
    if (data.type === 'visual') {
      // Only handle left mouse button - other buttons are for panning or context menu
      if (e.evt.button !== 0) {
        return;
      }

      // Check if we started a potential bbox draw
      if (this.$isBboxDrawing.get()) {
        const startCoord = this.$bboxStartCoord.get();

        // Check if we actually dragged by calculating from start position
        const cursorPos = this.manager.tool.$cursorPos.get();
        if (!cursorPos || !startCoord) {
          // Stop tracking even if we don't have valid coords
          this.$isBboxDrawing.set(false);
          this.$bboxStartCoord.set(null);
          return;
        }

        // Stop tracking (after we've used the values)
        this.$isBboxDrawing.set(false);
        this.$bboxStartCoord.set(null);

        const pixelRect = this.parent.transformer.$pixelRect.get();
        const parentPosition = addCoords(this.parent.state.position, pixelRect);
        const currentPoint = offsetCoord(cursorPos.relative, parentPosition);

        const dragWidth = Math.abs(currentPoint.x - startCoord.x);
        const dragHeight = Math.abs(currentPoint.y - startCoord.y);

        // Check if we actually dragged (moved more than threshold)
        if (dragWidth > 5 || dragHeight > 5) {
          // Get the final bbox dimensions from the rect's attributes
          const x = this.konva.bboxRect.x();
          const y = this.konva.bboxRect.y();
          // Get the actual dimensions, accounting for any scale
          const width = Math.max(1, this.konva.bboxRect.width() * this.konva.bboxRect.scaleX());
          const height = Math.max(1, this.konva.bboxRect.height() * this.konva.bboxRect.scaleY());

          // Reset scale to prevent accumulation issues
          this.konva.bboxRect.setAttrs({
            width: width,
            height: height,
            scaleX: 1,
            scaleY: 1,
          });

          // It was a drag - save the bbox
          this.$inputData.set({
            ...data,
            bbox: {
              x: x,
              y: y,
              width: width,
              height: height,
            },
          });

          // Show the transformer for resizing
          this.konva.bboxTransformer.visible(true);
          // Enable listening now that drawing is complete
          this.konva.bboxRect.listening(true);
        } else {
          // It was just a click, not a drag - add a point instead
          // Make sure existing bbox stays visible
          this.syncBboxVisibility();

          // Ignore if the stage is dragging/panning
          if (this.manager.stage.getIsDragging()) {
            return;
          }

          // Ignore if a point is being dragged
          if (this.$isDraggingPoint.get()) {
            return;
          }

          // Ignore if we are already processing
          if (this.$isProcessing.get()) {
            return;
          }

          if (!startCoord) {
            return;
          }

          const point = this.createPoint(startCoord, this.getPointType(e));
          const newPoints = [...data.points, point];
          this.$inputData.set({ ...data, points: newPoints });

          // Ensure bbox remains visible if it exists
          this.syncBboxVisibility();
        }
        return;
      }

      return;
    }

    // Handle prompt mode - nothing to do on pointer up
  };

  /**
   * Handles mouse/touch move for manual bbox dragging detection
   */
  onBboxDragMove = () => {
    const dragStart = this.$bboxDragStart.get();
    if (!dragStart) {
      return;
    }

    // If the stage is being dragged (e.g., with middle mouse), clear our bbox drag state
    if (this.manager.stage.getIsDragging()) {
      this.$bboxDragStart.set(null);
      this.$isBboxDragging.set(false);
      return;
    }

    // If we're already dragging, no need to check again
    if (this.$isBboxDragging.get()) {
      return;
    }

    const stage = this.manager.stage.konva.stage;
    const pointerPos = stage.getPointerPosition();
    if (!pointerPos) {
      return;
    }

    // Calculate the distance moved
    const dx = Math.abs(pointerPos.x - dragStart.x);
    const dy = Math.abs(pointerPos.y - dragStart.y);
    const distance = Math.sqrt(dx * dx + dy * dy);

    // If moved more than 5 pixels, start dragging
    if (distance > 5) {
      this.$isBboxDragging.set(true);
      // Enable dragging and start the drag programmatically
      this.konva.bboxRect.draggable(true);
      this.konva.bboxRect.startDrag();
    }
  };

  getPointType = (e: KonvaEventObject<PointerEvent>): SAMPointLabel => {
    let pointType = this.$pointType.get();
    if (e.evt.shiftKey) {
      pointType = pointType === 1 ? -1 : 1; // Invert the point type if shift is held
    }
    return pointType;
  };

  /**
   * Handles mouse/touch up for manual bbox dragging
   */
  onBboxDragEnd = (e: KonvaEventObject<PointerEvent>) => {
    // Only handle left mouse button - other buttons are for panning or context menu
    if (e.evt.button !== 0) {
      return;
    }

    const dragStart = this.$bboxDragStart.get();
    if (!dragStart) {
      return;
    }

    // Clear the drag state
    this.$bboxDragStart.set(null);

    // If we didn't drag, it was a click - allow point creation
    if (!this.$isBboxDragging.get()) {
      // Get the pointer position from the stage to create a point at the correct location
      const stage = this.manager.stage.konva.stage;
      const pointerPos = stage.getPointerPosition();
      if (pointerPos) {
        // Convert stage coordinates to relative coordinates
        const stageTransform = stage.getAbsoluteTransform().copy().invert();
        const relativePos = stageTransform.point(pointerPos);

        // Offset by parent position to get the correct point location
        const pixelRect = this.parent.transformer.$pixelRect.get();
        const parentPosition = addCoords(this.parent.state.position, pixelRect);
        const normalizedPoint = offsetCoord(relativePos, parentPosition);

        const data = this.$inputData.get();
        if (data.type === 'visual') {
          const point = this.createPoint(normalizedPoint, this.getPointType(e));
          const newPoints = [...data.points, point];
          this.$inputData.set({ ...data, points: newPoints });
        }
      }
    } else {
      // We did drag - disable dragging again for next time
      this.konva.bboxRect.draggable(false);
      // Update the bbox data after drag
      const data = this.$inputData.get();
      if (data.type === 'visual') {
        const x = this.konva.bboxRect.x();
        const y = this.konva.bboxRect.y();
        const width = this.konva.bboxRect.width();
        const height = this.konva.bboxRect.height();

        this.$inputData.set({
          ...data,
          bbox: {
            x: x,
            y: y,
            width: width,
            height: height,
          },
        });
      }
    }

    this.$isBboxDragging.set(false);
  };

  /**
   * Adds event listeners needed while segmenting the entity.
   */
  subscribe = () => {
    this.manager.stage.konva.stage.on('pointerdown', this.onStagePointerDown);
    this.manager.stage.konva.stage.on('pointermove', this.onStagePointerMove);
    this.manager.stage.konva.stage.on('pointerup', this.onStagePointerUp);

    // Add global listeners for bbox drag detection
    this.manager.stage.konva.stage.on('pointermove', this.onBboxDragMove);
    this.manager.stage.konva.stage.on('pointerup', this.onBboxDragEnd);

    this.subscriptions.add(() => {
      this.manager.stage.konva.stage.off('pointerdown', this.onStagePointerDown);
      this.manager.stage.konva.stage.off('pointermove', this.onStagePointerMove);
      this.manager.stage.konva.stage.off('pointerup', this.onStagePointerUp);
      this.manager.stage.konva.stage.off('pointermove', this.onBboxDragMove);
      this.manager.stage.konva.stage.off('pointerup', this.onBboxDragEnd);
    });

    // When we change the processing status, we should update the cursor style and the layer's listening status. For
    // example, when processing, we should disable listening on the layer so the user can't add more points, else we
    // should enable listening.
    this.subscriptions.add(
      this.$isProcessing.listen((isProcessing) => {
        this.syncCursorStyle();
        this.parent.konva.layer.listening(!isProcessing);
      })
    );

    // Scale the SAM points when the stage scale changes
    this.subscriptions.add(
      this.manager.stage.$stageAttrs.listen((stageAttrs, oldStageAttrs) => {
        if (areStageAttrsGonnaExplode(stageAttrs)) {
          return;
        }
        if (stageAttrs.scale !== oldStageAttrs.scale) {
          this.syncPointScales();
        }
      })
    );

    // When the input data changes, sync bbox visibility and process if autoProcess is enabled
    this.subscriptions.add(
      this.$inputData.listen((inputData) => {
        // Always sync bbox visibility when input data changes
        this.syncBboxVisibility();

        if (!hasInputData(inputData)) {
          return;
        }

        if (this.manager.stateApi.getSettings().autoProcess) {
          this.process();
        }
      })
    );

    // When the invert flag changes, process if autoProcess is enabled
    this.subscriptions.add(
      this.$invert.listen(() => {
        if (!hasInputData(this.$inputData.get())) {
          return;
        }

        if (this.manager.stateApi.getSettings().autoProcess) {
          this.process();
        }
      })
    );

    // When the model changes, process if autoProcess is enabled
    this.subscriptions.add(
      this.$model.listen(() => {
        if (!hasInputData(this.$inputData.get())) {
          return;
        }

        if (this.manager.stateApi.getSettings().autoProcess) {
          this.process();
        }
      })
    );

    // When auto-process is enabled, process the points if they have not been processed
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectAutoProcess, (autoProcess) => {
        if (!hasInputData(this.$inputData.get())) {
          return;
        }
        if (autoProcess) {
          this.process();
        }
      })
    );
  };

  /**
   * Removes event listeners used while segmenting the entity.
   */
  unsubscribe = () => {
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
  };

  /**
   * Starts the segmenting process.
   */
  start = () => {
    const segmentingAdapter = this.manager.stateApi.$segmentingAdapter.get();
    if (segmentingAdapter) {
      this.log.error(`Already segmenting an entity: ${segmentingAdapter.id}`);
      return;
    }
    this.log.trace('Starting segment anything');

    // Reset the module's state
    this.resetEphemeralState();
    this.$isSegmenting.set(true);

    // Update the konva group's position to match the parent entity
    const pixelRect = this.parent.transformer.$pixelRect.get();
    const position = addCoords(this.parent.state.position, pixelRect);
    this.konva.group.setAttrs(position);

    // Add the module's Konva group to the parent adapter's layer so it is rendered
    this.parent.konva.layer.add(this.konva.group);

    // Enable listening on the parent adapter's layer so the module can receive pointer events
    this.parent.konva.layer.listening(true);

    // Subscribe all listeners needed for segmenting (e.g. window pointerup, state listeners)
    this.subscribe();

    // Set the global segmenting adapter to this module
    this.manager.stateApi.$segmentingAdapter.set(this.parent);

    // Sync the cursor style to crosshair
    this.syncCursorStyle();
  };

  /**
   * Processes the SAM points to segment the entity, updating the module's state and rendering the mask.
   */
  processImmediate = async () => {
    if (!this.$isSegmenting.get()) {
      this.log.warn('Cannot process segmentation when not initialized');
      return;
    }

    if (this.$isProcessing.get()) {
      this.log.warn('Already processing');
      return;
    }

    const data = this.$inputData.get();
    const invert = this.$invert.get();
    const model = this.$model.get();

    if (!hasInputData(data)) {
      this.log.trace('No points to segment and no prompt provided');
      return;
    }

    const hash = stableHash({ inputData: getHashableInputData(data), invert, model });
    if (hash === this.$lastProcessedHash.get()) {
      this.log.trace('Already processed inputs');
      return;
    }

    this.$isProcessing.set(true);

    this.log.trace({ inputData: getHashableInputData(data), invert, model }, 'Segmenting');

    // Rasterize the entity in its current state
    const rect = this.parent.transformer.getRelativeRect();
    const rasterizeResult = await withResultAsync(() =>
      this.parent.renderer.rasterize({ rect, attrs: { filters: [], opacity: 1 } })
    );

    if (rasterizeResult.isErr()) {
      toast({ status: 'error', title: 'Failed to select object' });
      this.log.error({ error: serializeError(rasterizeResult.error) }, 'Error rasterizing entity');
      this.$isProcessing.set(false);
      return;
    }

    // Create an AbortController for the segmenting process
    const controller = new AbortController();
    this.abortController = controller;

    // Build the graph for segmenting the image, using the rasterized image DTO
    const { graph, outputNodeId } = CanvasSegmentAnythingModule.buildGraph(rasterizeResult.value, data, invert, model);

    // Run the graph and get the segmented image output
    const segmentResult = await withResultAsync(() =>
      this.manager.stateApi.runGraphAndReturnImageOutput({
        graph,
        outputNodeId,
        options: {
          prepend: true,
          signal: controller.signal,
        },
      })
    );

    // If there is an error, log it and bail out of this processing run
    if (segmentResult.isErr()) {
      this.log.error({ error: serializeError(segmentResult.error) }, 'Error segmenting');
      this.$isProcessing.set(false);
      // Clean up the abort controller as needed
      if (!this.abortController.signal.aborted) {
        this.abortController.abort();
      }
      this.abortController = null;
      return;
    }

    this.log.trace({ imageDTO: segmentResult.value }, 'Segmented');

    // Prepare the ephemeral image state
    const imageState = imageDTOToImageObject(segmentResult.value);
    this.$imageState.set(imageState);

    // Destroy any existing masked image and create a new one
    if (this.imageModule) {
      this.imageModule.destroy();
    }
    if (this.konva.maskTween) {
      this.konva.maskTween.destroy();
      this.konva.maskTween = null;
    }

    this.imageModule = new CanvasObjectImage(imageState, this);

    // Force update the masked image - after awaiting, the image will be rendered (in memory)
    await this.imageModule.update(imageState, true);

    // Update the compositing rect to match the image size
    this.konva.compositingRect.setAttrs({
      width: imageState.image.width,
      height: imageState.image.height,
      visible: true,
    });

    // Now we can add the masked image to the mask group. It will be rendered above the compositing rect, but should be
    // under it, so we will move the compositing rect to the top
    this.konva.maskGroup.add(this.imageModule.konva.group);
    this.konva.compositingRect.moveToTop();

    // Cache the group to ensure the mask is rendered correctly w/ opacity
    this.konva.maskGroup.cache();

    // Create a pulsing tween
    this.konva.maskTween = new Konva.Tween({
      node: this.konva.maskGroup,
      duration: 1,
      opacity: 0.4, // oscillate between this value and pre-tween opacity
      yoyo: true,
      repeat: Infinity,
      easing: Konva.Easings.EaseOut,
    });

    // Start the pulsing effect
    this.konva.maskTween.play();

    this.$lastProcessedHash.set(hash);

    // We are done processing (still segmenting though!)
    this.$isProcessing.set(false);

    // Clean up the abort controller as needed
    if (!this.abortController.signal.aborted) {
      this.abortController.abort();
    }
    this.abortController = null;
  };

  /**
   * Debounced version of processImmediate.
   */
  process = debounce(this.processImmediate, this.config.PROCESS_DEBOUNCE_MS);

  /**
   * Applies the segmented image to the entity, replacing the entity's objects with the masked image.
   */
  apply = () => {
    const imageState = this.$imageState.get();
    if (!imageState) {
      this.log.error('No image state to apply');
      return;
    }
    this.log.trace('Applying');

    // Rasterize the entity, replacing the objects with the masked image
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

    // Final cleanup and teardown, returning user to main canvas UI
    this.teardown();
  };

  /**
   * Saves the segmented image as a new entity of the given type.
   * @param type The type of entity to save the segmented image as.
   */
  saveAs = (type: CanvasEntityType) => {
    const imageState = this.$imageState.get();
    if (!imageState) {
      this.log.error('No image state to save as');
      return;
    }
    this.log.trace(`Saving as ${type}`);

    // Have the parent adopt the image module - this prevents a flash of the original layer content before the
    // segmented image is rendered
    if (this.imageModule) {
      this.parent.renderer.adoptObjectRenderer(this.imageModule);
    }

    // Create the new entity with the masked image as its only object
    const rect = this.parent.transformer.getRelativeRect();
    const arg = {
      overrides: {
        objects: [imageState],
        position: {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
        },
      },
      isSelected: true,
    };

    switch (type) {
      case 'raster_layer':
        this.manager.stateApi.addRasterLayer(arg);
        break;
      case 'control_layer':
        this.manager.stateApi.addControlLayer(arg);
        break;
      case 'inpaint_mask':
        this.manager.stateApi.addInpaintMask(arg);
        break;
      case 'regional_guidance':
        this.manager.stateApi.addRegionalGuidance(arg);
        break;
      default:
        assert<Equals<typeof type, never>>(false);
    }
  };

  /**
   * Resets the module (e.g. remove all points and the mask image).
   *
   * Does not cancel or otherwise complete the segmenting process.
   */
  reset = () => {
    this.log.trace('Resetting');
    this.resetEphemeralState();
  };

  setInputType = (type: PromptInputData['type'] | VisualInputData['type']) => {
    const data = this.$inputData.get();
    if (data.type === type) {
      return;
    }
    this.reset();
    if (type === 'prompt') {
      this.$inputData.set({ type: 'prompt', prompt: '' });
    } else {
      this.$inputData.set({ type: 'visual', points: [], bbox: null });
      // Hide bbox nodes when switching to visual mode (they'll be shown when drawing)
      this.konva.bboxRect.visible(false);
      this.konva.bboxTransformer.visible(false);
    }
  };

  /**
   * Cancels the segmenting process.
   */
  cancel = () => {
    this.log.trace('Canceling');
    // Reset the module's state and tear down, returning user to main canvas UI
    this.teardown();
  };

  /**
   * Performs teardown of the module. This shared logic is used for canceling and applying - when the segmenting is
   * complete and the module is deactivated.
   *
   * This method:
   * - Removes the module's main Konva node from the parent adapter's layer
   * - Removes segmenting event listeners (e.g. window pointerup)
   * - Resets the segmenting state
   * - Resets the global segmenting adapter
   */
  teardown = () => {
    this.unsubscribe();
    this.konva.group.remove();
    // The reset must be done _after_ unsubscribing from listeners, in case the listeners would otherwise react to
    // the reset. For example, if auto-processing is enabled and we reset the state, it may trigger processing.
    this.resetEphemeralState();
    this.$isSegmenting.set(false);
    this.manager.stateApi.$segmentingAdapter.set(null);
  };

  /**
   * Resets the module's ephemeral state. This shared logic is used for resetting, canceling, and applying.
   *
   * This method:
   * - Aborts any processing
   * - Destroys ephemeral Konva nodes
   * - Resets internal module state
   * - Resets non-ephemeral Konva nodes
   * - Clears the parent module's buffer
   */
  resetEphemeralState = () => {
    // First we need to bail out of any processing
    this.abortController?.abort();
    this.abortController = null;

    // Destroy ephemeral konva nodes
    const data = this.$inputData.get();
    if (data.type === 'visual') {
      // Destroy all points
      for (const point of data.points) {
        point.konva.circle.destroy();
      }

      // Hide bounding box nodes and reset drag state
      this.konva.bboxRect.visible(false);
      this.konva.bboxRect.draggable(false); // Ensure draggable is reset
      this.konva.bboxTransformer.visible(false);
      this.$isBboxDrawing.set(false);
      this.$bboxStartCoord.set(null);
      this.$bboxDragStart.set(null);
      this.$isBboxDragging.set(false);
    }

    // If the image module exists, and is a child of the group, destroy it. It might not be a child of the group if
    // the user has applied the segmented image and the image has been adopted by the parent entity.
    if (this.imageModule && this.imageModule.konva.group.parent === this.konva.group) {
      this.imageModule.destroy();
      this.imageModule = null;
    }
    if (this.konva.maskTween) {
      this.konva.maskTween.destroy();
      this.konva.maskTween = null;
    }

    // Empty internal module state - default to visual mode
    this.$inputData.set({ type: 'visual', points: [], bbox: null });
    this.$imageState.set(null);
    this.$pointType.set(1);
    this.$invert.set(false);
    this.$lastProcessedHash.set('');
    this.$isProcessing.set(false);

    // Reset non-ephemeral konva nodes
    this.konva.compositingRect.visible(false);
    this.konva.maskGroup.clearCache();
  };

  /**
   * Builds a graph for segmenting an image with the given image DTO.
   */
  static buildGraph = (
    { image_name }: ImageDTO,
    inputData: PromptInputData | VisualInputData,
    invert: boolean,
    model: SAMModel
  ): { graph: Graph; outputNodeId: string } => {
    const graph = new Graph(getPrefixedId('canvas_segment_anything'));

    const imagePrimitive = graph.addNode({
      id: getPrefixedId('image_primitive'),
      type: 'image',
      image: { image_name },
    });

    // For visual mode, we may have points, bbox, or both
    let pointLists = undefined;
    let boundingBoxes = undefined;

    if (inputData.type === 'visual') {
      // If we have points, add them
      if (inputData.points.length > 0) {
        pointLists = [{ points: getSAMPoints(inputData).map(({ x, y, label }) => ({ x, y, label })) }];
      }

      // If we have a bbox, add it
      if (inputData.bbox) {
        boundingBoxes = [
          {
            x_min: Math.round(inputData.bbox.x),
            y_min: Math.round(inputData.bbox.y),
            x_max: Math.round(inputData.bbox.x + inputData.bbox.width),
            y_max: Math.round(inputData.bbox.y + inputData.bbox.height),
          },
        ];
      }
    }

    const segmentAnything = graph.addNode({
      id: getPrefixedId('segment_anything'),
      type: 'segment_anything',
      model: model === 'SAM1' ? 'segment-anything-huge' : 'segment-anything-2-large',
      point_lists: pointLists,
      bounding_boxes: boundingBoxes,
      mask_filter: 'largest',
      apply_polygon_refinement: false,
    });

    graph.addEdge(imagePrimitive, 'image', segmentAnything, 'image');

    if (inputData.type === 'prompt') {
      const groundingDino = graph.addNode({
        id: getPrefixedId('grounding_dino'),
        type: 'grounding_dino',
        model: 'grounding-dino-base',
        prompt: inputData.prompt,
      });
      graph.addEdge(imagePrimitive, 'image', groundingDino, 'image');
      graph.addEdge(groundingDino, 'collection', segmentAnything, 'bounding_boxes');
    }

    // Apply the mask to the image, outputting an image w/ alpha transparency
    const applyMask = graph.addNode({
      id: getPrefixedId('apply_tensor_mask_to_image'),
      type: 'apply_tensor_mask_to_image',
      image: { image_name },
      invert,
    });
    graph.addEdge(segmentAnything, 'mask', applyMask, 'mask');

    return {
      graph,
      outputNodeId: applyMask.id,
    };
  };

  /**
   * Gets the color of a SAM point based on its label.
   */
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
    const data = this.$inputData.get();
    let inputData: JsonObject;
    if (data.type === 'prompt') {
      inputData = { type: 'prompt', prompt: data.prompt };
    } else {
      inputData = {
        type: 'visual',
        points: data.points.map(({ id, konva, label }) => ({
          id,
          label,
          circle: getKonvaNodeDebugAttrs(konva.circle),
        })),
        bbox: data.bbox || null,
      };
    }

    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      inputData,
      imageState: deepClone(this.$imageState.get()),
      imageModule: this.imageModule?.repr() ?? null,
      config: deepClone(this.config),
      $isSegmenting: this.$isSegmenting.get(),
      $lastProcessedHash: this.$lastProcessedHash.get(),
      $isProcessing: this.$isProcessing.get(),
      $pointType: this.$pointType.get(),
      $pointTypeString: this.$pointTypeString.get(),
      $isDraggingPoint: this.$isDraggingPoint.get(),
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
    if (this.abortController && !this.abortController.signal.aborted) {
      this.abortController.abort();
    }
    this.abortController = null;
    this.unsubscribe();
    this.konva.group.destroy();
  };
}
