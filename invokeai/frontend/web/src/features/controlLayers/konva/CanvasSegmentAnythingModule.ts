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
  getKonvaNodeDebugAttrs,
  getPrefixedId,
  offsetCoord,
  roundCoord,
} from 'features/controlLayers/konva/util';
import { selectAutoProcess } from 'features/controlLayers/store/canvasSettingsSlice';
import type {
  CanvasImageState,
  CanvasRenderableEntityType,
  Coordinate,
  RgbaColor,
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
import { debounce } from 'lodash-es';
import type { Atom } from 'nanostores';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';
import type { ImageDTO } from 'services/api/types';
import stableHash from 'stable-hash';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

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
   * The current input points. A listener is added to this atom to process the points when they change.
   */
  $points = atom<SAMPointState[]>([]);

  /**
   * Whether the module has points. This is a computed value based on $points.
   */
  $hasPoints = computed(this.$points, (points) => points.length > 0);

  /**
   * Whether the module should invert the mask image.
   */
  $invert = atom<boolean>(false);

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

    // Points should always be rendered above the mask group
    this.konva.group.add(this.konva.maskGroup);
    this.konva.group.add(this.konva.pointGroup);

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

      const newPoints = this.$points.get().filter((point) => point.id !== id);
      if (newPoints.length === 0) {
        this.resetEphemeralState();
      } else {
        this.$points.set(newPoints);
      }
    });

    circle.on('dragstart', () => {
      this.$isDraggingPoint.set(true);
    });

    circle.on('dragend', () => {
      const roundedCoord = roundCoord(circle.position());

      this.log.trace({ ...roundedCoord, label: SAM_POINT_LABEL_NUMBER_TO_STRING[label] }, 'Moved SAM point');
      this.$isDraggingPoint.set(false);

      const newPoints = this.$points.get().map((point) => {
        if (point.id === id) {
          return { ...point, coord: roundedCoord };
        }
        return point;
      });

      this.$points.set(newPoints);
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
  getSAMPoints = (): SAMPointWithId[] => {
    const points: SAMPointWithId[] = [];

    for (const { id, coord, label } of this.$points.get()) {
      points.push({
        id,
        x: coord.x,
        y: coord.y,
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
    const newPoints = [...this.$points.get(), point];
    this.$points.set(newPoints);
  };

  /**
   * Adds event listeners needed while segmenting the entity.
   */
  subscribe = () => {
    this.manager.stage.konva.stage.on('pointerup', this.onStagePointerUp);
    this.subscriptions.add(() => {
      this.manager.stage.konva.stage.off('pointerup', this.onStagePointerUp);
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

        if (this.manager.stateApi.getSettings().autoProcess) {
          this.process();
        }
      })
    );

    // When the invert flag changes, process if autoProcess is enabled
    this.subscriptions.add(
      this.$invert.listen(() => {
        if (this.$points.get().length === 0) {
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
        if (this.$points.get().length === 0) {
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

    const points = this.getSAMPoints();

    if (points.length === 0) {
      this.log.trace('No points to segment');
      return;
    }

    const invert = this.$invert.get();

    const hash = stableHash({ points, invert });
    if (hash === this.$lastProcessedHash.get()) {
      this.log.trace('Already processed points');
      return;
    }

    this.$isProcessing.set(true);

    this.log.trace({ points }, 'Segmenting');

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
    const { graph, outputNodeId } = CanvasSegmentAnythingModule.buildGraph(rasterizeResult.value, points, invert);

    // Run the graph and get the segmented image output
    const segmentResult = await withResultAsync(() =>
      this.manager.stateApi.runGraphAndReturnImageOutput({
        graph,
        outputNodeId,
        prepend: true,
        signal: controller.signal,
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
  saveAs = (type: CanvasRenderableEntityType) => {
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
    for (const point of this.$points.get()) {
      point.konva.circle.destroy();
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

    // Empty internal module state
    this.$points.set([]);
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
    points: SAMPointWithId[],
    invert: boolean
  ): { graph: Graph; outputNodeId: string } => {
    const graph = new Graph(getPrefixedId('canvas_segment_anything'));

    // TODO(psyche): When SAM2 is available in transformers, use it here
    // See: https://github.com/huggingface/transformers/pull/32317
    const segmentAnything = graph.addNode({
      id: getPrefixedId('segment_anything'),
      type: 'segment_anything',
      model: 'segment-anything-huge',
      image: { image_name },
      point_lists: [{ points: points.map(({ x, y, label }) => ({ x, y, label })) }],
      mask_filter: 'largest',
    });

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
