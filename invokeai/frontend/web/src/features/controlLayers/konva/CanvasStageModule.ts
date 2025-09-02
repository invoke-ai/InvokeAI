import type { Property } from 'csstype';
import { clamp } from 'es-toolkit/compat';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getKonvaNodeDebugAttrs, getPrefixedId, getRectUnion } from 'features/controlLayers/konva/util';
import type { Coordinate, Dimensions, Rect, StageAttrs } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';

type CanvasStageModuleConfig = {
  /**
   * The minimum (furthest-zoomed-in) scale of the canvas
   */
  MIN_SCALE: number;
  /**
   * The maximum (furthest-zoomed-out) scale of the canvas
   */
  MAX_SCALE: number;
  /**
   * The factor by which the canvas should be scaled when zooming in/out
   */
  SCALE_FACTOR: number;
  /**
   * The padding in pixels to use when fitting the layers to the stage.
   */
  FIT_LAYERS_TO_STAGE_PADDING_PX: number;
  /**
   * The snap points for the scale of the canvas.
   */
  SCALE_SNAP_POINTS: number[];
  /**
   * The tolerance for snapping the scale of the canvas, as a fraction of the scale.
   */
  SCALE_SNAP_TOLERANCE: number;
};

const DEFAULT_CONFIG: CanvasStageModuleConfig = {
  MIN_SCALE: 0.1,
  MAX_SCALE: 20,
  SCALE_FACTOR: 0.999,
  FIT_LAYERS_TO_STAGE_PADDING_PX: 48,
  SCALE_SNAP_POINTS: [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5],
  SCALE_SNAP_TOLERANCE: 0.02,
};

export class CanvasStageModule extends CanvasModuleBase {
  readonly type = 'stage';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  // State for scale snapping logic
  private _intendedScale: number = 1;
  private _activeSnapPoint: number | null = null;
  private _snapTimeout: number | null = null;
  private _lastScrollEventTimestamp: number | null = null;

  container: HTMLDivElement;
  konva: { stage: Konva.Stage };

  config: CanvasStageModuleConfig = DEFAULT_CONFIG;

  $stageAttrs = atom<StageAttrs>({
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    scale: 0,
  });
  $scale = computed(this.$stageAttrs, (attrs) => attrs.scale);

  subscriptions = new Set<() => void>();
  resizeObserver: ResizeObserver | null = null;

  constructor(container: HTMLDivElement, manager: CanvasManager) {
    super();
    this.id = getPrefixedId('stage');
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.container = container;
    this.konva = {
      stage: new Konva.Stage({
        id: getPrefixedId('konva_stage'),
        container,
      }),
    };

    // Initialize intended scale to the default stage scale
    this._intendedScale = this.konva.stage.scaleX();
  }

  setContainer = (container: HTMLDivElement) => {
    this.container = container;
    this.konva.stage.container(container);
    this.setResizeObserver();
  };

  setResizeObserver = () => {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    this.resizeObserver = new ResizeObserver(this.fitStageToContainer);
    this.resizeObserver.observe(this.container);
  };

  initialize = () => {
    this.log.debug('Initializing module');
    this.container.style.touchAction = 'none';
    this.container.style.userSelect = 'none';
    this.container.style.webkitUserSelect = 'none';
    this.konva.stage.container(this.container);
    this.setResizeObserver();
    this.fitStageToContainer();

    this.konva.stage.on('wheel', this.onStageMouseWheel);
    this.konva.stage.on('dragmove', this.onStageDragMove);
    this.konva.stage.on('dragend', this.onStageDragEnd);

    // Start dragging the stage when the middle mouse button is clicked. We do not need to listen for 'pointerdown' to
    // do cleanup - that is done in onStageDragEnd.
    this.konva.stage.on('pointerdown', this.onStagePointerDown);

    this.subscriptions.add(() => this.konva.stage.off('wheel', this.onStageMouseWheel));
    this.subscriptions.add(() => this.konva.stage.off('dragmove', this.onStageDragMove));
    this.subscriptions.add(() => this.konva.stage.off('dragend', this.onStageDragEnd));

    // Whenever the tool changes, we should stop dragging the stage. For example, user is MMB-dragging the stage, then
    // switches to the brush tool, we should stop dragging the stage.
    this.subscriptions.add(this.manager.tool.$tool.listen(this.stopDragging));
  };

  /**
   * Fits the stage to the container element.
   */
  fitStageToContainer = (): void => {
    this.log.trace('Fitting stage to container');
    const containerWidth = this.konva.stage.container().offsetWidth;
    const containerHeight = this.konva.stage.container().offsetHeight;

    // If the container has no size, the following calculations will be reaallll funky and bork the stage
    if (containerWidth === 0 || containerHeight === 0) {
      return;
    }

    // If the stage _had_ no size just before this function was called, that means we've just mounted the stage or
    // maybe un-hidden it. In that case, the user is about to see the stage for the first time, so we should fit the
    // layers to the stage. If we don't do this, the layers will not be centered.
    if (this.konva.stage.width() === 0 || this.konva.stage.height() === 0) {
      // This fit must happen before the stage size is set, else we can end up with a brief flash of an incorrectly
      // sized and scaled stage.
      this.fitLayersToStage({ animate: false, targetWidth: containerWidth, targetHeight: containerHeight });
    }

    this.konva.stage.width(containerWidth);
    this.konva.stage.height(containerHeight);
    this.syncStageAttrs();
  };

  /**
   * Fits the bbox to the stage. This will center the bbox and scale it to fit the stage with some padding.
   */
  fitBboxToStage = (options?: { animate?: boolean; targetWidth?: number; targetHeight?: number }): void => {
    const bbox = this.manager.stateApi.getBbox();
    if (!bbox) {
      this.log.warn('No bbox available, cannot fit to stage');
      return;
    }
    const { rect } = bbox;
    this.log.trace({ rect }, 'Fitting bbox to stage');
    this.fitRect(rect, options);
  };

  /**
   * Fits the visible canvas to the stage. This will center the canvas and scale it to fit the stage with some padding.
   */
  fitLayersToStage = (options?: { animate?: boolean; targetWidth?: number; targetHeight?: number }): void => {
    const rect = this.manager.compositor.getVisibleRectOfType();
    if (rect.width === 0 || rect.height === 0) {
      this.fitBboxToStage(options);
    } else {
      this.log.trace({ rect }, 'Fitting layers to stage');
      this.fitRect(rect, options);
    }
  };

  /**
   * Fits the bbox and layers to the stage. The union of the bbox and the visible layers will be centered and scaled
   * to fit the stage with some padding.
   */
  fitBboxAndLayersToStage = (options?: { animate?: boolean; targetWidth?: number; targetHeight?: number }): void => {
    const layersRect = this.manager.compositor.getVisibleRectOfType();
    const bbox = this.manager.stateApi.getBbox();
    if (!bbox) {
      this.log.warn('No bbox available, fitting layers only to stage');
      this.fitRect(layersRect, options);
      return;
    }
    const bboxRect = bbox.rect;
    const unionRect = getRectUnion(layersRect, bboxRect);
    this.log.trace({ bboxRect, layersRect, unionRect }, 'Fitting bbox and layers to stage');
    this.fitRect(unionRect, options);
  };

  /**
   * Fits a rectangle to the stage. The rectangle will be centered and scaled to fit the stage with some padding.
   *
   * The max scale is 1, but the stage can be scaled down to fit the rect.
   */
  fitRect = (rect: Rect, options?: { animate?: boolean; targetWidth?: number; targetHeight?: number }): void => {
    const size = this.getSize();
    const { animate, targetWidth, targetHeight } = {
      animate: true,
      targetWidth: size.width,
      targetHeight: size.height,
      ...options,
    };

    // If the stage has no size, we can't fit anything to it
    if (targetWidth === 0 || targetHeight === 0) {
      return;
    }

    const availableWidth = targetWidth - this.config.FIT_LAYERS_TO_STAGE_PADDING_PX * 2;
    const availableHeight = targetHeight - this.config.FIT_LAYERS_TO_STAGE_PADDING_PX * 2;

    // Make sure we don't accidentally set the scale to something nonsensical, like a negative number, 0 or something
    // outside the valid range
    const scale = this.constrainScale(
      Math.min(Math.min(availableWidth / rect.width, availableHeight / rect.height), 1)
    );
    const x = Math.floor(
      -rect.x * scale + this.config.FIT_LAYERS_TO_STAGE_PADDING_PX + (availableWidth - rect.width * scale) / 2
    );
    const y = Math.floor(
      -rect.y * scale + this.config.FIT_LAYERS_TO_STAGE_PADDING_PX + (availableHeight - rect.height * scale) / 2
    );

    // When fitting the stage, we update the intended scale and reset any active snap.
    this._intendedScale = scale;
    this._activeSnapPoint = null;

    if (animate) {
      const tween = new Konva.Tween({
        node: this.konva.stage,
        duration: 0.15,
        x,
        y,
        scaleX: scale,
        scaleY: scale,
        easing: Konva.Easings.EaseInOut,
        onUpdate: () => {
          this.syncStageAttrs();
        },
        onFinish: () => {
          this.syncStageAttrs();
          tween.destroy();
        },
      });
      tween.play();
    } else {
      this.konva.stage.setAttrs({
        x,
        y,
        scaleX: scale,
        scaleY: scale,
      });
      this.syncStageAttrs();
    }
  };

  /**
   * Gets the center of the stage in either absolute or relative coordinates
   * @param absolute Whether to return the center in absolute coordinates
   */
  getCenter = (absolute = false): Coordinate => {
    const scale = this.getScale();
    const { x, y } = this.getPosition();
    const { width, height } = this.getSize();

    const center = {
      x: (width / 2 - x) / scale,
      y: (height / 2 - y) / scale,
    };

    if (!absolute) {
      return center;
    }

    return this.konva.stage.getAbsoluteTransform().point(center);
  };

  /**
   * Constrains a scale to be within the valid range
   */
  constrainScale = (scale: number): number => {
    return clamp(scale, this.config.MIN_SCALE, this.config.MAX_SCALE);
  };

  /**
   * Programmatically sets the scale of the stage, overriding any active snapping.
   * If a center point is provided, the stage will zoom on that point.
   * @param scale The new scale to set.
   * @param center The center point for the zoom.
   */
  setScale = (scale: number, center?: Coordinate): void => {
    this.log.trace({ scale }, 'Programmatically setting scale');
    const newScale = this.constrainScale(scale);

    // When scale is set programmatically, update the intended scale and reset any active snap.
    this._intendedScale = newScale;
    this._activeSnapPoint = null;

    this._applyScale(newScale, center);
  };

  /**
   * Applies a scale to the stage, adjusting the position to keep the given center point stationary.
   * This internal method does NOT modify snapping state.
   */
  private _applyScale = (newScale: number, center?: Coordinate): void => {
    const oldScale = this.getScale();

    const _center = center ?? this.getCenter(true);
    const { x, y } = this.getPosition();

    const deltaX = (_center.x - x) / oldScale;
    const deltaY = (_center.y - y) / oldScale;

    const newX = _center.x - deltaX * newScale;
    const newY = _center.y - deltaY * newScale;

    this.konva.stage.setAttrs({
      x: newX,
      y: newY,
      scaleX: newScale,
      scaleY: newScale,
    });

    this.syncStageAttrs();
  };

  onStageMouseWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    if (this._snapTimeout !== null) {
      window.clearTimeout(this._snapTimeout);
    }

    if (e.evt.ctrlKey || e.evt.metaKey) {
      return;
    }

    // We need the absolute cursor position - not the scaled position
    const cursorPos = this.konva.stage.getPointerPosition();

    if (!cursorPos) {
      return;
    }

    // When wheeling on trackpad, e.evt.ctrlKey is true - in that case, let's reverse the direction
    const scrollAmount = e.evt.ctrlKey ? -e.evt.deltaY : e.evt.deltaY;

    const now = window.performance.now();
    const deltaT = this._lastScrollEventTimestamp === null ? Infinity : now - this._lastScrollEventTimestamp;
    this._lastScrollEventTimestamp = now;

    let dynamicScaleFactor = this.config.SCALE_FACTOR;

    if (deltaT > 300) {
      dynamicScaleFactor = this.config.SCALE_FACTOR + (1 - this.config.SCALE_FACTOR) / 2;
    } else if (deltaT < 300) {
      // Ensure dynamic scale factor stays below 1 to maintain zoom-out direction - if it goes over, we could end up
      // zooming in the wrong direction with small scroll amounts
      const maxScaleFactor = 0.9999;
      dynamicScaleFactor = Math.min(
        this.config.SCALE_FACTOR + (1 - this.config.SCALE_FACTOR) * (deltaT / 200),
        maxScaleFactor
      );
    }

    // Update the intended scale based on the last intended scale, creating a continuous zoom feel
    // Handle the sign explicitly to prevent direction reversal with small scroll amounts
    const scaleFactor =
      scrollAmount > 0
        ? dynamicScaleFactor ** Math.abs(scrollAmount)
        : (1 / dynamicScaleFactor) ** Math.abs(scrollAmount);
    const newIntendedScale = this._intendedScale * scaleFactor;
    this._intendedScale = this.constrainScale(newIntendedScale);

    // Pass control to the snapping logic
    this._updateScaleWithSnapping(cursorPos);

    this._snapTimeout = window.setTimeout(() => {
      // After a short delay, we can reset the intended scale to the current scale
      // This allows for continuous zooming without snapping back to the last snapped scale
      this._intendedScale = this.getScale();
    }, 300);
  };

  /**
   * Implements "sticky" snap logic.
   * - If not snapped, checks if the intended scale is close enough to a snap point to engage the snap.
   * - If snapped, checks if the intended scale has moved far enough away to break the snap.
   * - Applies the resulting scale to the stage.
   */
  private _updateScaleWithSnapping = (center: Coordinate) => {
    // If we are currently snapped, check if we should break out
    if (this._activeSnapPoint !== null) {
      const threshold = this._activeSnapPoint * this.config.SCALE_SNAP_TOLERANCE;
      if (Math.abs(this._intendedScale - this._activeSnapPoint) > threshold) {
        // User has scrolled far enough to break the snap
        this._activeSnapPoint = null;
        this._applyScale(this._intendedScale, center);
      } else {
        // Reset intended scale to prevent drift while snapped
        this._intendedScale = this._activeSnapPoint;
      }
      // Else, do nothing - we remain snapped at the current scale, creating a "dead zone"
      return;
    }

    // If we are not snapped, check if we should snap to a point
    for (const snapPoint of this.config.SCALE_SNAP_POINTS) {
      const threshold = snapPoint * this.config.SCALE_SNAP_TOLERANCE;
      if (Math.abs(this._intendedScale - snapPoint) < threshold) {
        // Engage the snap
        this._activeSnapPoint = snapPoint;
        this._applyScale(snapPoint, center);
        return;
      }
    }

    // If we are not snapping and not breaking a snap, just update to the intended scale
    this._applyScale(this._intendedScale, center);
  };

  onStagePointerDown = (e: KonvaEventObject<PointerEvent>) => {
    // If the middle mouse button is clicked and we are not already dragging, start dragging the stage
    if (e.evt.button === 1) {
      this.startDragging();
    }
  };

  /**
   * Forcibly starts dragging the stage. This is useful when you want to start dragging the stage programmatically.
   */
  startDragging = () => {
    // First make sure the stage is draggable
    this.setIsDraggable(true);

    // Then start dragging the stage if it's not already being dragged
    if (!this.getIsDragging()) {
      this.konva.stage.startDrag();
    }

    // And render the tool to update the cursor
    this.manager.tool.render();
  };

  /**
   * Stops dragging the stage. This is useful when you want to stop dragging the stage programmatically.
   */
  stopDragging = () => {
    // Now that we have stopped the current drag event, we may need to revert the stage's draggable status, depending
    // on the current tool
    this.setIsDraggable(this.manager.tool.$tool.get() === 'view');

    // Stop dragging the stage if it's being dragged
    if (this.getIsDragging()) {
      this.konva.stage.stopDrag();
    }

    // And render the tool to update the cursor
    this.manager.tool.render();
  };

  onStageDragMove = (e: KonvaEventObject<MouseEvent>) => {
    if (e.target !== this.konva.stage) {
      return;
    }

    this.syncStageAttrs();
  };

  onStageDragEnd = (e: KonvaEventObject<DragEvent>) => {
    if (e.target !== this.konva.stage) {
      return;
    }
    // Do some cleanup when the stage is no longer being dragged
    this.stopDragging();
  };

  /**
   * Gets the scale of the stage. The stage is always scaled uniformly in x and y.
   */
  getScale = (): number => {
    // The stage is never scaled differently in x and y
    return this.konva.stage.scaleX();
  };

  /**
   * Gets the position of the stage.
   */
  getPosition = (): Coordinate => {
    return this.konva.stage.position();
  };

  /**
   * Gets the size of the stage.
   */
  getSize(): Dimensions {
    return this.konva.stage.size();
  }

  /**
   * Unscales a value by the current stage scale. For example, if the stage scale is 5, and you want to unscale 10
   * pixels, would be scaled to 10px / 5 = 2 pixels.
   */
  unscale = (value: number): number => {
    return value / this.getScale();
  };

  setCursor = (cursor: Property.Cursor) => {
    if (this.container.style.cursor === cursor) {
      return;
    }
    this.container.style.cursor = cursor;
  };

  syncStageAttrs = (overrides?: Partial<StageAttrs>) => {
    this.$stageAttrs.set({
      x: this.konva.stage.x(),
      y: this.konva.stage.y(),
      width: this.konva.stage.width(),
      height: this.konva.stage.height(),
      scale: this.konva.stage.scaleX(),
      ...overrides,
    });
  };

  setIsDraggable = (isDraggable: boolean) => {
    this.konva.stage.draggable(isDraggable);
  };

  getIsDragging = () => {
    return this.konva.stage.isDragging();
  };

  addLayer = (layer: Konva.Layer) => {
    this.konva.stage.add(layer);
  };

  /**
   * Gets the rectangle of the stage in the absolute coordinates. This can be used to draw a rect that covers the
   * entire stage.
   * @param snapToInteger Whether to snap the values to integers. Default is true. This is useful when the stage is
   * scaled in such a way that the absolute (screen) coordinates or dimensions are not integers.
   */
  getScaledStageRect = (snapToInteger: boolean = true): Rect => {
    const { x, y } = this.getPosition();
    const { width, height } = this.getSize();
    const scale = this.getScale();
    return {
      x: snapToInteger ? Math.floor(-x / scale) : -x / scale,
      y: snapToInteger ? Math.floor(-y / scale) : -y / scale,
      width: snapToInteger ? Math.ceil(width / scale) : width / scale,
      height: snapToInteger ? Math.ceil(height / scale) : height / scale,
    };
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      config: this.config,
      $stageAttrs: this.$stageAttrs.get(),
      konva: {
        stage: getKonvaNodeDebugAttrs(this.konva.stage),
      },
    };
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    this.konva.stage.destroy();
  };
}
