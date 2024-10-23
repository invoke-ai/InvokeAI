import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import { withResultAsync } from 'common/util/result';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import { addCoords, getKonvaNodeDebugAttrs, getPrefixedId, offsetCoord } from 'features/controlLayers/konva/util';
import { selectAutoProcess } from 'features/controlLayers/store/canvasSettingsSlice';
import type {
  CanvasImageState,
  Coordinate,
  RgbaColor,
  SAMPoint,
  SAMPointLabel,
  SAMPointLabelString,
} from 'features/controlLayers/store/types';
import { SAM_POINT_LABEL_NUMBER_TO_STRING } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { debounce } from 'lodash-es';
import type { Atom } from 'nanostores';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';
import type { ImageDTO } from 'services/api/types';

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
  MASK_COLOR: { r: 0, g: 200, b: 200, a: 0.5 }, // cyan with 50% opacity
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

  /**
   * Whether the module is currently segmenting an entity.
   */
  $isSegmenting = atom<boolean>(false);

  /**
   * Whether the current set of points has been processed.
   */
  $hasProcessed = atom<boolean>(false);

  /**
   * Whether the module is currently processing the points.
   */
  $isProcessing = atom<boolean>(false);

  /**
   * The type of point to create when segmenting. This is a number representation of the SAMPointLabel enum.
   */
  $pointType = atom<SAMPointLabel>(1);

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
  imageState: CanvasImageState | null = null;

  /**
   * The current input points.
   */
  $points = atom<SAMPointState[]>([]);

  /**
   * Whether the module has points.
   */
  $hasPoints = computed(this.$points, (points) => points.length > 0);

  /**
   * The masked image object, if it exists.
   */
  maskedImage: CanvasObjectImage | null = null;

  /**
   * The Konva nodes for the module.
   */
  konva: {
    /**
     * The main Konva group node for the module.
     */
    group: Konva.Group;
    /**
     * The Konva group node for the SAM points.
     *
     * This is a child of the main group node, rendered above the mask group.
     */
    pointGroup: Konva.Group;
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
  };

  KONVA_CIRCLE_NAME = `${this.type}:circle`;
  KONVA_GROUP_NAME = `${this.type}:group`;
  KONVA_POINT_GROUP_NAME = `${this.type}:point_group`;
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
      maskGroup: new Konva.Group({ name: this.KONVA_MASK_GROUP_NAME }),
      compositingRect: new Konva.Rect({
        name: this.KONVA_COMPOSITING_RECT_NAME,
        fill: rgbaColorToString(this.config.MASK_COLOR),
        globalCompositeOperation: 'source-atop',
        listening: false,
        strokeEnabled: false,
        perfectDrawEnabled: false,
        visible: false,
      }),
    };

    // Mask group is below the point group
    this.konva.group.add(this.konva.maskGroup);
    this.konva.group.add(this.konva.pointGroup);

    // Compositing rect is added to the mask group - will also be above the mask image, but that doesn't get created
    // until after processing
    this.konva.maskGroup.add(this.konva.compositingRect);

    this.subscriptions.add(
      this.$isProcessing.listen((isProcessing) => {
        this.syncCursorStyle();
        if (this.$isSegmenting.get()) {
          this.parent.konva.layer.listening(!isProcessing);
        }
      })
    );

    // Scale the SAM points when the stage scale changes
    this.subscriptions.add(
      this.manager.stage.$stageAttrs.listen((stageAttrs, oldStageAttrs) => {
        if (stageAttrs.scale !== oldStageAttrs.scale) {
          this.syncPointScales();
        }
      })
    );

    // When the points change, process them if autoProcess is enabled
    this.subscriptions.add(
      this.$points.listen((points) => {
        if (points.length === 0) {
          return;
        }
        if (this.manager.stateApi.getSettings().autoProcess && this.$isSegmenting.get()) {
          this.process();
        }
      })
    );

    // When auto-process is enabled, process the points if they have not been processed
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectAutoProcess, (autoProcess) => {
        if (this.$points.get().length === 0) {
          return;
        }
        if (autoProcess && this.$isSegmenting.get() && !this.$hasProcessed.get()) {
          this.process();
        }
      })
    );
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

    const circle = new Konva.Circle({
      name: this.KONVA_CIRCLE_NAME,
      x: Math.round(coord.x),
      y: Math.round(coord.y),
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
      // This event should not bubble up to the parent, stage or any other nodes
      e.cancelBubble = true;
      circle.destroy();
      this.$points.set(this.$points.get().filter((point) => point.id !== id));
      this.$hasProcessed.set(false);
    });

    circle.on('dragstart', () => {
      this.$isDraggingPoint.set(true);
    });

    circle.on('dragend', () => {
      this.$isDraggingPoint.set(false);
      // Point has changed!
      this.$hasProcessed.set(false);
      this.$points.notify();
      this.log.trace(
        { x: Math.round(circle.x()), y: Math.round(circle.y()), label: SAM_POINT_LABEL_NUMBER_TO_STRING[label] },
        'Moved SAM point'
      );
    });

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

  /**
   * Synchronizes the scales of the SAM points to the stage scale.
   *
   * SAM points are always the same size, regardless of the stage scale.
   */
  syncPointScales = () => {
    const radius = this.manager.stage.unscale(this.config.SAM_POINT_RADIUS);
    const borderWidth = this.manager.stage.unscale(this.config.SAM_POINT_BORDER_WIDTH);
    for (const point of this.$points.get()) {
      point.konva.circle.radius(radius);
      point.konva.circle.strokeWidth(borderWidth);
    }
  };

  /**
   * Gets the SAM points in the format expected by the segment-anything API. The x and y values are rounded to integers.
   */
  getSAMPoints = (): SAMPoint[] => {
    const points: SAMPoint[] = [];

    for (const { konva, label } of this.$points.get()) {
      points.push({
        // Pull out and round the x and y values from Konva
        x: Math.round(konva.circle.x()),
        y: Math.round(konva.circle.y()),
        label,
      });
    }

    return points;
  };

  /**
   * Handles the pointerup event on the stage. This is used to add a SAM point to the module.
   */
  onStagePointerUp = (e: KonvaEventObject<PointerEvent>) => {
    // Only handle left-clicks
    if (e.evt.button !== 0) {
      return;
    }

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

    // Ignore if the cursor is not within the stage (should never happen)
    const cursorPos = this.manager.tool.$cursorPos.get();
    if (!cursorPos) {
      return;
    }

    // We need to offset the cursor position by the parent entity's position + pixel rect to get the correct position
    const pixelRect = this.parent.transformer.$pixelRect.get();
    const parentPosition = addCoords(this.parent.state.position, pixelRect);

    // Normalize the cursor position to the parent entity's position
    const normalizedPoint = offsetCoord(cursorPos.relative, parentPosition);

    // Create a SAM point at the normalized position
    const point = this.createPoint(normalizedPoint, this.$pointType.get());
    this.$points.set([...this.$points.get(), point]);

    // Mark the module as having _not_ processed the points now that they have changed
    this.$hasProcessed.set(false);
  };

  /**
   * Adds Konva stage event listeners for segmenting the entity.
   */
  addStageEventListeners = () => {
    this.manager.stage.konva.stage.on('pointerup', this.onStagePointerUp);
  };

  /**
   * Removes Konva stage event listeners for segmenting the entity.
   */
  removeStageEventListeners = () => {
    this.manager.stage.konva.stage.off('pointerup', this.onStagePointerUp);
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

    // Set up the segmenting event listeners (e.g. window pointerup)
    this.addStageEventListeners();

    // Set the global segmenting adapter to this module
    this.manager.stateApi.$segmentingAdapter.set(this.parent);

    // Sync the cursor style to crosshair
    this.syncCursorStyle();
  };

  /**
   * Processes the SAM points to segment the entity, updating the module's state and rendering the mask.
   */
  processImmediate = async () => {
    if (this.$isProcessing.get()) {
      this.log.warn('Already processing');
      return;
    }

    const points = this.getSAMPoints();

    if (points.length === 0) {
      this.log.trace('No points to segment');
      return;
    }

    this.$isProcessing.set(true);

    this.log.trace({ points }, 'Segmenting');
    const rect = this.parent.transformer.getRelativeRect();

    const rasterizeResult = await withResultAsync(() =>
      this.parent.renderer.rasterize({ rect, attrs: { filters: [], opacity: 1 } })
    );

    if (rasterizeResult.isErr()) {
      this.log.error({ error: serializeError(rasterizeResult.error) }, 'Error rasterizing entity');
      this.$isProcessing.set(false);
      return;
    }

    const controller = new AbortController();
    this.abortController = controller;

    const { graph, outputNodeId } = this.buildGraph(rasterizeResult.value);

    const segmentResult = await withResultAsync(() =>
      this.manager.stateApi.runGraphAndReturnImageOutput({
        graph,
        outputNodeId,
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

  process = debounce(this.processImmediate, this.config.PROCESS_DEBOUNCE_MS);

  /**
   * Applies the segmented image to the entity.
   */
  apply = () => {
    if (!this.$hasProcessed.get()) {
      this.log.error('Cannot apply unprocessed points');
      return;
    }
    const imageState = this.imageState;
    if (!imageState) {
      this.log.error('No image state to apply');
      return;
    }
    this.log.trace('Applying');
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
    this.resetEphemeralState();
    this.teardown();
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

  /**
   * Cancels the segmenting process.
   */
  cancel = () => {
    this.log.trace('Canceling');
    this.resetEphemeralState();
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
    this.konva.group.remove();
    this.removeStageEventListeners();
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
    for (const point of this.$points.get()) {
      point.konva.circle.destroy();
    }
    if (this.maskedImage) {
      this.maskedImage.destroy();
    }

    // Empty internal module state
    this.$points.set([]);
    this.imageState = null;
    this.$pointType.set(1);
    this.$hasProcessed.set(false);
    this.$isProcessing.set(false);

    // Reset non-ephemeral konva nodes
    this.konva.compositingRect.visible(false);
    this.konva.maskGroup.clearCache();

    // The parent module's buffer should be reset & forcibly sync the cache
    this.parent.bufferRenderer.clearBuffer();
    this.parent.renderer.syncKonvaCache(true);
  };

  /**
   * Builds a graph for segmenting an image with the given image DTO.
   */
  buildGraph = ({ image_name }: ImageDTO): { graph: Graph; outputNodeId: string } => {
    const graph = new Graph(getPrefixedId('canvas_segment_anything'));

    // TODO(psyche): When SAM2 is available in transformers, use it here
    // See: https://github.com/huggingface/transformers/pull/32317
    const segmentAnything = graph.addNode({
      id: getPrefixedId('segment_anything'),
      type: 'segment_anything',
      model: 'segment-anything-huge',
      image: { image_name },
      point_lists: [{ points: this.getSAMPoints() }],
      mask_filter: 'largest',
    });

    // Apply the mask to the image, outputting an image w/ alpha transparency
    const applyMask = graph.addNode({
      id: getPrefixedId('apply_tensor_mask_to_image'),
      type: 'apply_tensor_mask_to_image',
      image: { image_name },
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
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      points: this.$points.get().map(({ id, konva, label }) => ({
        id,
        label,
        circle: getKonvaNodeDebugAttrs(konva.circle),
      })),
      imageState: deepClone(this.imageState),
      maskedImage: this.maskedImage?.repr(),
      config: deepClone(this.config),
      $isSegmenting: this.$isSegmenting.get(),
      $hasProcessed: this.$hasProcessed.get(),
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
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.removeStageEventListeners();
    this.konva.group.destroy();
  };
}
