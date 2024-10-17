import { Mutex } from 'async-mutex';
import { withResultAsync } from 'common/util/result';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import {
  canvasToImageData,
  getEmptyRect,
  getKonvaNodeDebugAttrs,
  getPrefixedId,
} from 'features/controlLayers/konva/util';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import type { Coordinate, Rect, RectWithRotation } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { GroupConfig } from 'konva/lib/Group';
import { clamp, debounce, get } from 'lodash-es';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';
import { assert } from 'tsafe';

type CanvasEntityTransformerConfig = {
  /**
   * The debounce time in milliseconds for calculating the rect of the parent entity
   */
  RECT_CALC_DEBOUNCE_MS: number;
  /**
   * The padding around the scaling transform anchors for hit detection
   */
  ANCHOR_HIT_PADDING: number;
  /**
   * The padding around the parent entity when drawing the rect outline
   */
  OUTLINE_PADDING: number;
  /**
   * The color of the rect outline
   */
  OUTLINE_COLOR: string;
  /**
   * The fill color of the scaling transform anchors
   */
  SCALE_ANCHOR_FILL_COLOR: string;
  /**
   * The stroke color of the scaling transform anchors
   */
  SCALE_ANCHOR_STROKE_COLOR: string;
  /**
   * The corner radius ratio of the scaling transform anchors
   */
  SCALE_ANCHOR_CORNER_RADIUS_RATIO: number;
  /**
   * The stroke width of the scaling transform anchors
   */
  SCALE_ANCHOR_STROKE_WIDTH: number;
  /**
   * The size of the scaling transform anchors
   */
  SCALE_ANCHOR_SIZE: number;
  /**
   * The fill color of the rotation transform anchor
   */
  ROTATE_ANCHOR_FILL_COLOR: string;
  /**
   * The stroke color of the rotation transform anchor
   */
  ROTATE_ANCHOR_STROKE_COLOR: string;
  /**
   * The size (height/width) of the rotation transform anchor
   */
  ROTATE_ANCHOR_SIZE: number;
};

const DEFAULT_CONFIG: CanvasEntityTransformerConfig = {
  RECT_CALC_DEBOUNCE_MS: 300,
  ANCHOR_HIT_PADDING: 10,
  OUTLINE_PADDING: 0,
  OUTLINE_COLOR: 'hsl(200 76% 50% / 1)', // invokeBlue.500
  SCALE_ANCHOR_FILL_COLOR: 'hsl(200 76% 50% / 1)', // invokeBlue.500
  SCALE_ANCHOR_STROKE_COLOR: 'hsl(200 76% 77% / 1)', // invokeBlue.200
  SCALE_ANCHOR_CORNER_RADIUS_RATIO: 0.5,
  SCALE_ANCHOR_STROKE_WIDTH: 2,
  SCALE_ANCHOR_SIZE: 8,
  ROTATE_ANCHOR_FILL_COLOR: 'hsl(200 76% 95% / 1)', // invokeBlue.50
  ROTATE_ANCHOR_STROKE_COLOR: 'hsl(200 76% 40% / 1)', // invokeBlue.700
  ROTATE_ANCHOR_SIZE: 12,
};

export class CanvasEntityTransformer extends CanvasModuleBase {
  readonly type = 'entity_transformer';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityAdapter;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasEntityTransformerConfig = DEFAULT_CONFIG;

  /**
   * The rect of the parent, _including_ transparent regions, **relative to the parent's position**. To get the rect
   * relative to the _stage_, add the parent's position.
   *
   * It is calculated via Konva's getClientRect method, which is fast but includes transparent regions.
   *
   * This rect is relative _to the parent's position_, not the stage.
   */
  $nodeRect = atom<Rect>(getEmptyRect());

  /**
   * The rect of the parent, _excluding_ transparent regions, **relative to the parent's position**. To get the rect
   * relative to the _stage_, add the parent's position.
   *
   * If the parent's nodes have no possibility of transparent regions, this will be calculated the same way as nodeRect.
   *
   * If the parent's nodes may have transparent regions, this will be calculated manually by rasterizing the parent and
   * checking the pixel data.
   */
  $pixelRect = atom<Rect>(getEmptyRect());

  /**
   * Whether the transformer is currently calculating the rect of the parent.
   */
  $isPendingRectCalculation = atom<boolean>(true);

  /**
   * A set of subscriptions that should be cleaned up when the transformer is destroyed.
   */
  subscriptions: Set<() => void> = new Set();

  /**
   * Whether the transformer is currently transforming the entity.
   */
  $isTransforming = atom<boolean>(false);

  /**
   * The current interaction mode of the transformer:
   * - 'all': The entity can be moved, resized, and rotated.
   * - 'drag': The entity can be moved.
   * - 'off': The transformer is not interactable.
   */
  $interactionMode = atom<'all' | 'drag' | 'off'>('off');

  /**
   * Whether dragging is enabled. Dragging is enabled in both 'all' and 'drag' interaction modes.
   */
  $isDragEnabled = atom<boolean>(false);

  /**
   * Whether transforming is enabled. Transforming is enabled only in 'all' interaction mode.
   */
  $isTransformEnabled = atom<boolean>(false);

  /**
   * Whether the transformer is currently processing (rasterizing and uploading) the transformed entity.
   */
  $isProcessing = atom(false);

  /**
   * Whether the transformer is currently in silent mode. In silent mode, the transform operation should not show any
   * visual feedback.
   *
   * This is set every time a transform is started.
   *
   * This is used for transform operations like directly fitting the entity to the bbox, which should not show the
   * transform controls, Transform react component or have any other visual feedback. The transform should just happen
   * silently.
   */
  $silentTransform = atom(false);

  /**
   * A mutex to prevent concurrent operations.
   *
   * The mutex is locked during transformation and during rect calculations which are handled in a web worker.
   */
  transformMutex = new Mutex();

  konva: {
    transformer: Konva.Transformer;
    proxyRect: Konva.Rect;
    outlineRect: Konva.Rect;
  };

  constructor(parent: CanvasEntityTransformer['parent']) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.konva = {
      outlineRect: new Konva.Rect({
        listening: false,
        draggable: false,
        name: `${this.type}:outline_rect`,
        stroke: this.config.OUTLINE_COLOR,
        perfectDrawEnabled: false,
        hitStrokeWidth: 0,
      }),
      transformer: new Konva.Transformer({
        name: `${this.type}:transformer`,
        // Visibility and listening are managed via activate() and deactivate()
        visible: false,
        listening: false,
        // Rotation is allowed
        rotateEnabled: true,
        // When dragging a transform anchor across either the x or y axis, the nodes will be flipped across the axis
        flipEnabled: true,
        // Transforming will allow free aspect ratio only when shift is held
        keepRatio: true,
        shiftBehavior: 'inverted',
        // The padding is the distance between the transformer bbox and the nodes
        padding: this.config.OUTLINE_PADDING,
        // This is `invokeBlue.400`
        stroke: this.config.OUTLINE_COLOR,
        anchorFill: this.config.SCALE_ANCHOR_FILL_COLOR,
        anchorStroke: this.config.SCALE_ANCHOR_STROKE_COLOR,
        anchorStrokeWidth: this.config.SCALE_ANCHOR_STROKE_WIDTH,
        anchorSize: this.config.SCALE_ANCHOR_SIZE,
        anchorCornerRadius: this.config.SCALE_ANCHOR_SIZE * this.config.SCALE_ANCHOR_CORNER_RADIUS_RATIO,
        // This function is called for each anchor to style it (and do anything else you might want to do).
        anchorStyleFunc: this.anchorStyleFunc,
        anchorDragBoundFunc: this.anchorDragBoundFunc,
        boundBoxFunc: this.boxBoundFunc,
      }),
      proxyRect: new Konva.Rect({
        name: `${this.type}:proxy_rect`,
        listening: false,
        draggable: true,
      }),
    };

    this.konva.transformer.on('transform', this.syncObjectGroupWithProxyRect);
    this.konva.transformer.on('transformend', this.snapProxyRectToPixelGrid);
    this.konva.proxyRect.on('dragmove', this.onDragMove);
    this.konva.proxyRect.on('dragend', this.onDragEnd);

    // When the stage scale changes, we may need to re-scale some of the transformer's components. For example,
    // the bbox outline should always be 1 screen pixel wide, so we need to update its stroke width.
    this.subscriptions.add(
      this.manager.stage.$stageAttrs.listen((newVal, oldVal) => {
        if (newVal.scale !== oldVal.scale) {
          this.syncScale();
        }
      })
    );

    // While the user holds shift, we want to snap rotation to 45 degree increments. Listen for the shift key state
    // and update the snap angles accordingly.
    this.subscriptions.add(
      this.manager.stateApi.$shiftKey.listen((newVal) => {
        this.konva.transformer.rotationSnaps(newVal ? [0, 45, 90, 135, 180, 225, 270, 315] : []);
      })
    );

    // When the selected tool changes, we need to update the transformer's interaction state.
    this.subscriptions.add(this.manager.tool.$tool.listen(this.syncInteractionState));

    // When the selected entity changes, we need to update the transformer's interaction state.
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectSelectedEntityIdentifier, this.syncInteractionState)
    );

    /**
     * When the canvas global state changes, we need to update the transformer's interaction state. This implies
     * a change to staging or some other global state that affects the transformer.
     */
    this.subscriptions.add(this.manager.$isBusy.listen(this.syncInteractionState));

    this.parent.konva.layer.add(this.konva.outlineRect);
    this.parent.konva.layer.add(this.konva.proxyRect);
    this.parent.konva.layer.add(this.konva.transformer);
  }

  initialize = () => {
    this.log.debug('Initializing module');
    this.syncInteractionState();
  };

  anchorStyleFunc = (anchor: Konva.Rect): void => {
    // Give the rotater special styling
    if (anchor.hasName('rotater')) {
      anchor.setAttrs({
        height: this.config.ROTATE_ANCHOR_SIZE,
        width: this.config.ROTATE_ANCHOR_SIZE,
        cornerRadius: this.config.ROTATE_ANCHOR_SIZE * this.config.SCALE_ANCHOR_CORNER_RADIUS_RATIO,
        fill: this.config.ROTATE_ANCHOR_FILL_COLOR,
        stroke: this.config.SCALE_ANCHOR_FILL_COLOR,
        offsetX: this.config.ROTATE_ANCHOR_SIZE / 2,
        offsetY: this.config.ROTATE_ANCHOR_SIZE / 2,
      });
    }
    // Add some padding to the hit area of the anchors
    anchor.hitFunc((context) => {
      context.beginPath();
      context.rect(
        -this.config.ANCHOR_HIT_PADDING,
        -this.config.ANCHOR_HIT_PADDING,
        anchor.width() + this.config.ANCHOR_HIT_PADDING * 2,
        anchor.height() + this.config.ANCHOR_HIT_PADDING * 2
      );
      context.closePath();
      context.fillStrokeShape(anchor);
    });
  };

  anchorDragBoundFunc = (oldPos: Coordinate, newPos: Coordinate) => {
    // The anchorDragBoundFunc callback puts constraints on the movement of the transformer anchors, which in
    // turn constrain the transformation. It is called on every anchor move. We'll use this to snap the anchors
    // to the nearest pixel.

    // If we are rotating, no need to do anything - just let the rotation happen.
    if (this.konva.transformer.getActiveAnchor() === 'rotater') {
      return newPos;
    }

    // If the user is not holding shift, the transform is retaining aspect ratio. It's not possible to snap to the grid
    // in this case, because that would change the aspect ratio. So, we only snap to the grid when shift is held.
    const gridSize = this.manager.stateApi.$shiftKey.get() ? this.manager.stateApi.getGridSize() : 1;

    // We need to snap the anchor to the selected grid size, but the positions provided to this callback are absolute,
    // scaled coordinates. They need to be converted to stage coordinates, snapped, then converted back to absolute
    // before returning them.
    const stageScale = this.manager.stage.getScale();
    const stagePos = this.manager.stage.getPosition();

    // Unscale and snap the coordinate.
    const targetX = roundToMultiple(newPos.x / stageScale, gridSize);
    const targetY = roundToMultiple(newPos.y / stageScale, gridSize);

    // The stage may be offset by fraction of the grid snap size. To ensure the anchor snaps to the grid, we need to
    // calculate that offset and add it back to the target position.

    // Calculate the offset. It's the remainder of the stage position divided by the scale * grid snap value in pixels.
    const scaledOffsetX = stagePos.x % (stageScale * gridSize);
    const scaledOffsetY = stagePos.y % (stageScale * gridSize);

    // Unscale the target position and add the offset to get the absolute position for this anchor.
    const scaledTargetX = targetX * stageScale + scaledOffsetX;
    const scaledTargetY = targetY * stageScale + scaledOffsetY;

    return { x: scaledTargetX, y: scaledTargetY };
  };

  boxBoundFunc = (oldBoundBox: RectWithRotation, newBoundBox: RectWithRotation) => {
    // Bail if we are not rotating, we don't need to do anything.
    if (this.konva.transformer.getActiveAnchor() !== 'rotater') {
      return newBoundBox;
    }

    // This transform constraint operates on the bounding box of the transformer. This box has x, y, width, and
    // height in stage coordinates, and rotation in radians. This can be used to snap the transformer rotation to
    // the nearest 45 degrees when shift is held.
    if (this.manager.stateApi.$shiftKey.get()) {
      if (Math.abs(newBoundBox.rotation % (Math.PI / 4)) > 0) {
        return oldBoundBox;
      }
    }

    return newBoundBox;
  };

  /**
   * Snaps the proxy rect to the nearest pixel, syncing the object group with the proxy rect.
   */
  snapProxyRectToPixelGrid = () => {
    // Called on mouse up on an anchor. We'll do some final snapping to ensure the transformer is pixel-perfect.

    // Snap the position to the nearest pixel.
    const x = this.konva.proxyRect.x();
    const y = this.konva.proxyRect.y();
    const snappedX = Math.round(x);
    const snappedY = Math.round(y);

    // The transformer doesn't modify the width and height. It only modifies scale. We'll need to apply the scale to
    // the width and height, round them to the nearest pixel, and finally calculate a new scale that will result in
    // the snapped width and height.
    const width = this.konva.proxyRect.width();
    const height = this.konva.proxyRect.height();
    const scaleX = this.konva.proxyRect.scaleX();
    const scaleY = this.konva.proxyRect.scaleY();

    // Determine the target width and height, rounded to the nearest pixel. Must be >= 1. Because the scales can be
    // negative, we need to take the absolute value of the width and height.
    const targetWidth = Math.max(Math.abs(Math.round(width * scaleX)), 1);
    const targetHeight = Math.max(Math.abs(Math.round(height * scaleY)), 1);

    // Calculate the scale we need to use to get the target width and height. Restore the sign of the scales.
    const snappedScaleX = (targetWidth / width) * Math.sign(scaleX);
    const snappedScaleY = (targetHeight / height) * Math.sign(scaleY);

    // Update interaction rect and object group attributes.
    this.konva.proxyRect.setAttrs({
      x: snappedX,
      y: snappedY,
      scaleX: snappedScaleX,
      scaleY: snappedScaleY,
    });

    this.syncObjectGroupWithProxyRect();
  };

  /**
   * Fits the entity to the bbox using the "fill" strategy.
   */
  fitToBboxFill = () => {
    if (!this.$isTransformEnabled.get()) {
      this.log.warn(
        'Cannot fit to bbox contain when transform is disabled. Did you forget to call `await adapter.transformer.startTransform()`?'
      );
      return;
    }
    const { rect } = this.manager.stateApi.getBbox();
    const scaleX = rect.width / this.konva.proxyRect.width();
    const scaleY = rect.height / this.konva.proxyRect.height();
    this.konva.proxyRect.setAttrs({
      x: rect.x,
      y: rect.y,
      scaleX,
      scaleY,
      rotation: 0,
    });
    this.syncObjectGroupWithProxyRect();
  };

  /**
   * Fits the entity to the bbox using the "contain" strategy.
   */
  fitToBboxContain = () => {
    if (!this.$isTransformEnabled.get()) {
      this.log.warn(
        'Cannot fit to bbox contain when transform is disabled. Did you forget to call `await adapter.transformer.startTransform()`?'
      );
      return;
    }
    const { rect } = this.manager.stateApi.getBbox();
    const gridSize = this.manager.stateApi.getGridSize();
    const width = this.konva.proxyRect.width();
    const height = this.konva.proxyRect.height();
    const scaleX = rect.width / width;
    const scaleY = rect.height / height;

    // "contain" means that the entity should be scaled to fit within the bbox, but it should not exceed the bbox.
    const scale = Math.min(scaleX, scaleY);

    // Center the shape within the bounding box
    const offsetX = (rect.width - width * scale) / 2;
    const offsetY = (rect.height - height * scale) / 2;

    this.konva.proxyRect.setAttrs({
      x: clamp(roundToMultiple(rect.x + offsetX, gridSize), rect.x, rect.x + rect.width),
      y: clamp(roundToMultiple(rect.y + offsetY, gridSize), rect.y, rect.y + rect.height),
      scaleX: scale,
      scaleY: scale,
      rotation: 0,
    });
    this.syncObjectGroupWithProxyRect();
  };

  /**
   * Fits the entity to the bbox using the "cover" strategy.
   */
  fitToBboxCover = () => {
    if (!this.$isTransformEnabled.get()) {
      this.log.warn(
        'Cannot fit to bbox contain when transform is disabled. Did you forget to call `await adapter.transformer.startTransform()`?'
      );
      return;
    }
    const { rect } = this.manager.stateApi.getBbox();
    const gridSize = this.manager.stateApi.getGridSize();
    const width = this.konva.proxyRect.width();
    const height = this.konva.proxyRect.height();
    const scaleX = rect.width / width;
    const scaleY = rect.height / height;

    // "cover" is the same as "contain", but we choose the larger scale to cover the shape
    const scale = Math.max(scaleX, scaleY);

    // Center the shape within the bounding box
    const offsetX = (rect.width - width * scale) / 2;
    const offsetY = (rect.height - height * scale) / 2;

    this.konva.proxyRect.setAttrs({
      x: roundToMultiple(rect.x + offsetX, gridSize),
      y: roundToMultiple(rect.y + offsetY, gridSize),
      scaleX: scale,
      scaleY: scale,
      rotation: 0,
    });
    this.syncObjectGroupWithProxyRect();
  };

  onDragMove = () => {
    // Snap the interaction rect to the grid
    const gridSize = this.manager.stateApi.getGridSize();
    this.konva.proxyRect.x(roundToMultiple(this.konva.proxyRect.x(), gridSize));
    this.konva.proxyRect.y(roundToMultiple(this.konva.proxyRect.y(), gridSize));

    // The bbox should be updated to reflect the new position of the interaction rect, taking into account its padding
    // and border
    const padding = this.manager.stage.unscale(this.config.OUTLINE_PADDING);
    this.konva.outlineRect.setAttrs({
      x: this.konva.proxyRect.x() - padding,
      y: this.konva.proxyRect.y() - padding,
    });

    // The object group is translated by the difference between the interaction rect's new and old positions (which is
    // stored as this.pixelRect)
    this.parent.renderer.konva.objectGroup.setAttrs({
      x: this.konva.proxyRect.x(),
      y: this.konva.proxyRect.y(),
    });
  };

  onDragEnd = () => {
    if (this.$isTransforming.get()) {
      // If we are transforming the entity, we should not push the new position to the state. This will trigger a
      // re-render of the entity and bork the transformation.
      return;
    }

    const pixelRect = this.$pixelRect.get();

    const position = {
      x: this.konva.proxyRect.x() - pixelRect.x,
      y: this.konva.proxyRect.y() - pixelRect.y,
    };

    this.log.trace({ position }, 'Position changed');
    this.manager.stateApi.setEntityPosition({ entityIdentifier: this.parent.entityIdentifier, position });
  };

  syncObjectGroupWithProxyRect = () => {
    this.parent.renderer.konva.objectGroup.setAttrs({
      x: this.konva.proxyRect.x(),
      y: this.konva.proxyRect.y(),
      scaleX: this.konva.proxyRect.scaleX(),
      scaleY: this.konva.proxyRect.scaleY(),
      rotation: this.konva.proxyRect.rotation(),
    });
  };

  /**
   * Updates the transformer's visual components to match the parent entity's position and bounding box.
   * @param position The position of the parent entity
   * @param bbox The bounding box of the parent entity
   */
  update = (position: Coordinate, bbox: Rect) => {
    const onePixel = this.manager.stage.unscale(1);
    const bboxPadding = this.manager.stage.unscale(this.config.OUTLINE_PADDING);

    this.konva.outlineRect.setAttrs({
      x: position.x + bbox.x - bboxPadding,
      y: position.y + bbox.y - bboxPadding,
      width: bbox.width + bboxPadding * 2,
      height: bbox.height + bboxPadding * 2,
      strokeWidth: onePixel,
    });
    this.konva.proxyRect.setAttrs({
      x: position.x + bbox.x,
      y: position.y + bbox.y,
      width: bbox.width,
      height: bbox.height,
    });
  };

  /**
   * Syncs the transformer's interaction state with the application and entity's states. This is called when the entity
   * is selected or deselected, or when the user changes the selected tool.
   */
  syncInteractionState = () => {
    this.log.trace('Syncing interaction state');

    if (this.manager.$isBusy.get() && !this.$isTransforming.get()) {
      // The canvas is busy, we can't interact with the transformer
      this.parent.konva.layer.listening(false);
      this._setInteractionMode('off');
      return;
    }

    // Not all entities have a filterer - only raster layer and control layer adapters
    if (this.parent.filterer?.$isFiltering.get()) {
      // May not interact with the entity when the filter is active
      this.parent.konva.layer.listening(false);
      this._setInteractionMode('off');
      return;
    }

    if (this.manager.stateApi.$isTransforming.get() && !this.$isTransforming.get()) {
      // If another entity is being transformed, we can't interact with this transformer
      this.parent.konva.layer.listening(false);
      this._setInteractionMode('off');
      return;
    }

    const pixelRect = this.$pixelRect.get();
    const isPendingRectCalculation = this.$isPendingRectCalculation.get();

    if (isPendingRectCalculation || pixelRect.width === 0 || pixelRect.height === 0) {
      // If the rect is being calculated, or if the rect has no width or height, we can't interact with the transformer
      this.parent.konva.layer.listening(false);
      this._setInteractionMode('off');
      return;
    }

    const tool = this.manager.tool.$tool.get();
    const isSelected = this.manager.stateApi.getIsSelected(this.parent.id);

    if (this.parent.$isEmpty.get()) {
      // The layer is totally empty, we can just disable the layer
      this.parent.konva.layer.listening(false);
      this._setInteractionMode('off');
      return;
    }

    if (isSelected && !this.$isTransforming.get() && tool === 'move') {
      // We are moving this layer, it must be listening
      this.parent.konva.layer.listening(true);
      this._setInteractionMode('drag');
      return;
    }

    if (isSelected && this.$isTransforming.get()) {
      // When transforming, we want the stage to still be movable if the view tool is selected. If the transformer is
      // active, it will interrupt the stage drag events. So we should disable listening when the view tool is selected.
      if (tool === 'view') {
        this.parent.konva.layer.listening(false);
        this._setInteractionMode('off');
      } else {
        this.parent.konva.layer.listening(true);
        this._setInteractionMode('all');
      }
    } else {
      // The layer is not selected, or we are using a tool that doesn't need the layer to be listening - disable interaction stuff
      this.parent.konva.layer.listening(false);
      this._setInteractionMode('off');
    }
  };

  /**
   * Updates the transformer's scale. This is called when the stage is scaled.
   */
  syncScale = () => {
    const onePixel = this.manager.stage.unscale(1);
    const bboxPadding = this.manager.stage.unscale(this.config.OUTLINE_PADDING);

    this.konva.outlineRect.setAttrs({
      x: this.konva.proxyRect.x() - bboxPadding,
      y: this.konva.proxyRect.y() - bboxPadding,
      width: this.konva.proxyRect.width() * this.konva.proxyRect.scaleX() + bboxPadding * 2,
      height: this.konva.proxyRect.height() * this.konva.proxyRect.scaleY() + bboxPadding * 2,
      strokeWidth: onePixel,
    });
    this.konva.transformer.forceUpdate();
  };

  /**
   * Starts the transformation of the entity.
   *
   * This method will asynchronously acquire a mutex to prevent concurrent operations. If you need to perform an
   * operation after the transformation is started, you should await this method.
   *
   * @param arg Options for starting the transformation
   * @param arg.silent Whether the transformation should be silent. If silent, the transform controls will not be shown,
   * so you _must_ call `applyTransform` or `stopTransform` to complete the transformation.
   *
   * @example
   * ```ts
   * await adapter.transformer.startTransform({ silent: true });
   * adapter.transformer.fitToBboxContain();
   * await adapter.transformer.applyTransform();
   * ```
   */
  startTransform = async (arg?: { silent: boolean }) => {
    const transformingAdapter = this.manager.stateApi.$transformingAdapter.get();
    if (transformingAdapter) {
      assert(false, `Already transforming an entity: ${transformingAdapter.id}`);
    }
    // This will be released when the transformation is stopped
    await this.transformMutex.acquire();
    this.log.debug('Starting transform');
    const { silent } = { silent: false, ...arg };
    this.$silentTransform.set(silent);
    this.$isTransforming.set(true);
    this.manager.stateApi.$transformingAdapter.set(this.parent);
    this.syncInteractionState();
  };

  /**
   * Applies the transformation of the entity.
   */
  applyTransform = async () => {
    if (!this.$isTransforming.get()) {
      this.log.warn(
        'Cannot apply transform when not transforming. Did you forget to call `await adapter.transformer.startTransform()`?'
      );
      return;
    }
    this.log.debug('Applying transform');
    this.$isProcessing.set(true);
    this._setInteractionMode('off');
    const rect = this.getRelativeRect();
    const rasterizeResult = await withResultAsync(() =>
      this.parent.renderer.rasterize({
        rect,
        replaceObjects: true,
        ignoreCache: true,
        attrs: { opacity: 1, filters: [] },
      })
    );
    if (rasterizeResult.isErr()) {
      this.log.error({ error: serializeError(rasterizeResult.error) }, 'Failed to rasterize entity');
    }
    this.requestRectCalculation();
    this.stopTransform();
  };

  resetTransform = () => {
    this.resetScale();
    this.updatePosition();
    this.updateBbox();
  };

  /**
   * Stops the transformation of the entity. If the transformation is in progress, the entity will be reset to its
   * original state.
   */
  stopTransform = () => {
    this.log.debug('Stopping transform');

    this.$isTransforming.set(false);

    // Reset the transform of the the entity. We've either replaced the transformed objects with a rasterized image, or
    // canceled a transformation. In either case, the scale should be reset.
    this.resetTransform();
    this.syncInteractionState();
    this.manager.stateApi.$transformingAdapter.set(null);
    this.$isProcessing.set(false);
    this.transformMutex.release();
  };

  /**
   * Resets the scale of the transformer and the entity.
   * When the entity is transformed, it's scale and rotation are modified by the transformer. After canceling or applying
   * a transformation, the scale and rotation should be reset to the original values.
   */
  resetScale = () => {
    const attrs = {
      scaleX: 1,
      scaleY: 1,
      rotation: 0,
    };
    this.parent.renderer.konva.objectGroup.setAttrs(attrs);
    this.parent.bufferRenderer.konva.group.setAttrs(attrs);
    this.konva.outlineRect.setAttrs(attrs);
    this.konva.proxyRect.setAttrs(attrs);
  };

  /**
   * Updates the position of the transformer and the entity.
   * @param arg The position to update to. If omitted, the parent's last stored position will be used.
   */
  updatePosition = (arg?: { position: Coordinate }) => {
    this.log.trace('Updating position');
    const position = get(arg, 'position', this.parent.state.position);

    const pixelRect = this.$pixelRect.get();
    const groupAttrs: Partial<GroupConfig> = {
      x: position.x + pixelRect.x,
      y: position.y + pixelRect.y,
      offsetX: pixelRect.x,
      offsetY: pixelRect.y,
    };
    this.parent.renderer.konva.objectGroup.setAttrs(groupAttrs);
    this.parent.bufferRenderer.konva.group.setAttrs(groupAttrs);

    this.update(position, pixelRect);
  };

  /**
   * Sets the transformer to a specific interaction mode. This internal method shouldn't be used. Instead, use
   * `syncInteractionState` to update the transformer's interaction state.
   *
   * @param interactionMode The mode to set the transformer to. The transformer can be in one of three modes:
   * - 'all': The entity can be moved, resized, and rotated.
   * - 'drag': The entity can be moved.
   * - 'off': The transformer is not interactable.
   */
  _setInteractionMode = (interactionMode: 'all' | 'drag' | 'off') => {
    this.$interactionMode.set(interactionMode);
    if (interactionMode === 'drag') {
      this._enableDrag();
      this._disableTransform();
      this._showBboxOutline();
    } else if (interactionMode === 'all') {
      this._enableDrag();
      this._enableTransform();
      this._hideBboxOutline();
    } else if (interactionMode === 'off') {
      this._disableDrag();
      this._disableTransform();
      this._hideBboxOutline();
    }
  };

  updateBbox = () => {
    const nodeRect = this.$nodeRect.get();
    const pixelRect = this.$pixelRect.get();

    this.log.trace({ nodeRect, pixelRect }, 'Updating bbox');

    if (this.$isPendingRectCalculation.get()) {
      this.syncInteractionState();
      return;
    }

    // If the bbox has no width or height, that means the layer is fully transparent. This can happen if it is only
    // eraser lines, fully clipped brush lines or if it has been fully erased.
    if (pixelRect.width === 0 || pixelRect.height === 0) {
      // If the layer already has no objects, we don't need to reset the entity state. This would cause a push to the
      // undo stack and clear the redo stack.
      if (this.parent.renderer.hasObjects()) {
        this.manager.stateApi.resetEntity({ entityIdentifier: this.parent.entityIdentifier });
        this.syncInteractionState();
      }
    } else {
      this.syncInteractionState();
      this.update(this.parent.state.position, pixelRect);
      const groupAttrs: Partial<GroupConfig> = {
        x: this.parent.state.position.x + pixelRect.x,
        y: this.parent.state.position.y + pixelRect.y,
        offsetX: pixelRect.x,
        offsetY: pixelRect.y,
      };
      this.parent.renderer.konva.objectGroup.setAttrs(groupAttrs);
      this.parent.bufferRenderer.konva.group.setAttrs(groupAttrs);
    }
  };

  calculateRect = debounce(() => {
    this.log.debug('Calculating bbox');

    const canvas = this.parent.getCanvas();

    if (!this.parent.renderer.hasObjects()) {
      this.log.trace('No objects, resetting bbox');
      this.$nodeRect.set(getEmptyRect());
      this.$pixelRect.set(getEmptyRect());
      this.parent.$canvasCache.set(canvas);
      this.$isPendingRectCalculation.set(false);
      this.updateBbox();
      this.transformMutex.release();
      return;
    }

    const rect = this.parent.renderer.konva.objectGroup.getClientRect({ skipTransform: true });

    if (!this.parent.renderer.needsPixelBbox()) {
      this.$nodeRect.set({ ...rect });
      this.$pixelRect.set({ ...rect });
      this.log.trace({ nodeRect: this.$nodeRect.get(), pixelRect: this.$pixelRect.get() }, 'Got bbox from client rect');
      this.parent.$canvasCache.set(canvas);
      this.$isPendingRectCalculation.set(false);
      this.updateBbox();
      this.transformMutex.release();
      return;
    }

    // We have eraser strokes - we must calculate the bbox using pixel data
    const imageData = canvasToImageData(canvas);
    this.manager.worker.requestBbox(
      { buffer: imageData.data.buffer, width: imageData.width, height: imageData.height },
      (extents) => {
        if (extents) {
          const { minX, minY, maxX, maxY } = extents;
          this.$nodeRect.set({ ...rect });
          this.$pixelRect.set({
            x: Math.round(rect.x) + minX,
            y: Math.round(rect.y) + minY,
            width: maxX - minX,
            height: maxY - minY,
          });
        } else {
          this.$nodeRect.set(getEmptyRect());
          this.$pixelRect.set(getEmptyRect());
        }
        this.log.trace(
          { nodeRect: this.$nodeRect.get(), pixelRect: this.$pixelRect.get(), extents },
          `Got bbox from worker`
        );
        this.parent.$canvasCache.set(canvas);
        this.$isPendingRectCalculation.set(false);
        this.updateBbox();
        this.transformMutex.release();
      }
    );
  }, this.config.RECT_CALC_DEBOUNCE_MS);

  requestRectCalculation = async () => {
    // This will be released when the rect calculation is complete
    await this.transformMutex.acquire();
    this.$isPendingRectCalculation.set(true);
    this.syncInteractionState();
    this.calculateRect();
  };

  // TODO(psyche): After resetting an entity, this can return stale data...
  getRelativeRect = (): Rect => {
    return this.konva.proxyRect.getClientRect({ relativeTo: this.parent.konva.layer });
  };

  _enableTransform = () => {
    this.$isTransformEnabled.set(true);
    this.konva.transformer.visible(true);
    this.konva.transformer.listening(true);
    this.konva.transformer.nodes([this.konva.proxyRect]);
  };

  _disableTransform = () => {
    this.$isTransformEnabled.set(false);
    this.konva.transformer.visible(false);
    this.konva.transformer.listening(false);
    this.konva.transformer.nodes([]);
  };

  _enableDrag = () => {
    this.$isDragEnabled.set(true);
    this.konva.proxyRect.visible(true);
    this.konva.proxyRect.listening(true);
  };

  _disableDrag = () => {
    this.$isDragEnabled.set(false);
    this.konva.proxyRect.visible(false);
    this.konva.proxyRect.listening(false);
  };

  _showBboxOutline = () => {
    this.konva.outlineRect.visible(true);
  };

  _hideBboxOutline = () => {
    this.konva.outlineRect.visible(false);
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      config: this.config,
      $nodeRect: this.$nodeRect.get(),
      $pixelRect: this.$pixelRect.get(),
      $isPendingRectCalculation: this.$isPendingRectCalculation.get(),
      $isTransforming: this.$isTransforming.get(),
      $interactionMode: this.$interactionMode.get(),
      $isDragEnabled: this.$isDragEnabled.get(),
      $isTransformEnabled: this.$isTransformEnabled.get(),
      $isProcessing: this.$isProcessing.get(),
      konva: {
        transformer: getKonvaNodeDebugAttrs(this.konva.transformer),
        proxyRect: getKonvaNodeDebugAttrs(this.konva.proxyRect),
        outlineRect: getKonvaNodeDebugAttrs(this.konva.outlineRect),
      },
    };
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.konva.outlineRect.destroy();
    this.konva.transformer.destroy();
    this.konva.proxyRect.destroy();
  };
}
