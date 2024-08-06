import type { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import { getEmptyRect, getPrefixedId } from 'features/controlLayers/konva/util';
import type { Coordinate, GetLoggingContext, Rect } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { debounce, get } from 'lodash-es';
import type { Logger } from 'roarr';

/**
 * The CanvasTransformer class is responsible for managing the transformation of a canvas entity:
 * - Moving
 * - Resizing
 * - Rotating
 *
 * It renders an outline when dragging and resizing the entity, with transform anchors for resizing and rotation.
 */
export class CanvasTransformer {
  static TYPE = 'entity_transformer';
  static KONVA_TRANSFORMER_NAME = `${CanvasTransformer.TYPE}:transformer`;
  static KONVA_PROXY_RECT_NAME = `${CanvasTransformer.TYPE}:proxy_rect`;
  static KONVA_OUTLINE_RECT_NAME = `${CanvasTransformer.TYPE}:outline_rect`;

  static STROKE_COLOR = 'hsl(200 76% 50% / 1)'; // invokeBlue.500
  static ANCHOR_FILL_COLOR = CanvasTransformer.STROKE_COLOR;
  static ANCHOR_STROKE_COLOR = 'hsl(200 76% 77% / 1)'; // invokeBlue.200
  static RESIZE_ANCHOR_SIZE = 8;
  static ROTATE_ANCHOR_FILL_COLOR = 'hsl(200 76% 95% / 1)'; // invokeBlue.50
  static ROTATE_ANCHOR_STROKE_COLOR = 'hsl(200 76% 40% / 1)'; // invokeBlue.700
  static ROTATE_ANCHOR_SIZE = 12;
  static ANCHOR_CORNER_RADIUS_RATIO = 0.5;
  static ANCHOR_STROKE_WIDTH = 2;
  static ANCHOR_HIT_PADDING = 10;

  id: string;
  parent: CanvasLayerAdapter | CanvasMaskAdapter;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  /**
   * The rect of the parent, _including_ transparent regions.
   * It is calculated via Konva's getClientRect method, which is fast but includes transparent regions.
   */
  nodeRect = getEmptyRect();

  /**
   * The rect of the parent, _excluding_ transparent regions.
   * If the parent's nodes have no possibility of transparent regions, this will be calculated the same way as nodeRect.
   * If the parent's nodes may have transparent regions, this will be calculated manually by rasterizing the parent and
   * checking the pixel data.
   */
  pixelRect = getEmptyRect();

  /**
   * Whether the transformer is currently calculating the rect of the parent.
   */
  isPendingRectCalculation: boolean = false;

  /**
   * A set of subscriptions that should be cleaned up when the transformer is destroyed.
   */
  subscriptions: Set<() => void> = new Set();

  /**
   * Whether the transformer is currently transforming the entity.
   */
  isTransforming: boolean = false;

  /**
   * The current interaction mode of the transformer:
   * - 'all': The entity can be moved, resized, and rotated.
   * - 'drag': The entity can be moved.
   * - 'off': The transformer is not interactable.
   */
  interactionMode: 'all' | 'drag' | 'off' = 'off';

  /**
   * Whether dragging is enabled. Dragging is enabled in both 'all' and 'drag' interaction modes.
   */
  isDragEnabled: boolean = false;

  /**
   * Whether transforming is enabled. Transforming is enabled only in 'all' interaction mode.
   */
  isTransformEnabled: boolean = false;

  konva: {
    transformer: Konva.Transformer;
    proxyRect: Konva.Rect;
    outlineRect: Konva.Rect;
  };

  constructor(parent: CanvasLayerAdapter | CanvasMaskAdapter) {
    this.id = getPrefixedId(CanvasTransformer.TYPE);
    this.parent = parent;
    this.manager = parent.manager;

    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.konva = {
      outlineRect: new Konva.Rect({
        listening: false,
        draggable: false,
        name: CanvasTransformer.KONVA_OUTLINE_RECT_NAME,
        stroke: CanvasTransformer.STROKE_COLOR,
        perfectDrawEnabled: false,
        strokeHitEnabled: false,
      }),
      transformer: new Konva.Transformer({
        name: CanvasTransformer.KONVA_TRANSFORMER_NAME,
        // Visibility and listening are managed via activate() and deactivate()
        visible: false,
        listening: false,
        // Rotation is allowed
        rotateEnabled: true,
        // When dragging a transform anchor across either the x or y axis, the nodes will be flipped across the axis
        flipEnabled: true,
        // Transforming will retain aspect ratio only when shift is held
        keepRatio: false,
        // The padding is the distance between the transformer bbox and the nodes
        padding: this.manager.getTransformerPadding(),
        // This is `invokeBlue.400`
        stroke: CanvasTransformer.STROKE_COLOR,
        anchorFill: CanvasTransformer.ANCHOR_FILL_COLOR,
        anchorStroke: CanvasTransformer.ANCHOR_STROKE_COLOR,
        anchorStrokeWidth: CanvasTransformer.ANCHOR_STROKE_WIDTH,
        anchorSize: CanvasTransformer.RESIZE_ANCHOR_SIZE,
        anchorCornerRadius: CanvasTransformer.RESIZE_ANCHOR_SIZE * CanvasTransformer.ANCHOR_CORNER_RADIUS_RATIO,
        // This function is called for each anchor to style it (and do anything else you might want to do).
        anchorStyleFunc: (anchor) => {
          // Give the rotater special styling
          if (anchor.hasName('rotater')) {
            anchor.setAttrs({
              height: CanvasTransformer.ROTATE_ANCHOR_SIZE,
              width: CanvasTransformer.ROTATE_ANCHOR_SIZE,
              cornerRadius: CanvasTransformer.ROTATE_ANCHOR_SIZE * CanvasTransformer.ANCHOR_CORNER_RADIUS_RATIO,
              fill: CanvasTransformer.ROTATE_ANCHOR_FILL_COLOR,
              stroke: CanvasTransformer.ANCHOR_FILL_COLOR,
              offsetX: CanvasTransformer.ROTATE_ANCHOR_SIZE / 2,
              offsetY: CanvasTransformer.ROTATE_ANCHOR_SIZE / 2,
            });
          }
          // Add some padding to the hit area of the anchors
          anchor.hitFunc((context) => {
            context.beginPath();
            context.rect(
              -CanvasTransformer.ANCHOR_HIT_PADDING,
              -CanvasTransformer.ANCHOR_HIT_PADDING,
              anchor.width() + CanvasTransformer.ANCHOR_HIT_PADDING * 2,
              anchor.height() + CanvasTransformer.ANCHOR_HIT_PADDING * 2
            );
            context.closePath();
            context.fillStrokeShape(anchor);
          });
        },
        // TODO(psyche): The konva Vector2D type is is apparently not compatible with the JSONObject type that the log
        // function expects. The in-house Coordinate type is functionally the same - `{x: number; y: number}` - and
        // TypeScript is happy with it.
        anchorDragBoundFunc: (oldPos: Coordinate, newPos: Coordinate) => {
          // The anchorDragBoundFunc callback puts constraints on the movement of the transformer anchors, which in
          // turn constrain the transformation. It is called on every anchor move. We'll use this to snap the anchors
          // to the nearest pixel.

          // If we are rotating, no need to do anything - just let the rotation happen.
          if (this.konva.transformer.getActiveAnchor() === 'rotater') {
            return newPos;
          }

          // We need to snap the anchor to the nearest pixel, but the positions provided to this callback are absolute,
          // scaled coordinates. They need to be converted to stage coordinates, snapped, then converted back to absolute
          // before returning them.
          const stageScale = this.manager.getStageScale();
          const stagePos = this.manager.getStagePosition();

          // Unscale and round the target position to the nearest pixel.
          const targetX = Math.round(newPos.x / stageScale);
          const targetY = Math.round(newPos.y / stageScale);

          // The stage may be offset a fraction of a pixel. To ensure the anchor snaps to the nearest pixel, we need to
          // calculate that offset and add it back to the target position.

          // Calculate the offset. It's the remainder of the stage position divided by the scale * desired grid size. In
          // this case, the grid size is 1px. For example, if we wanted to snap to the nearest 8px, the calculation would
          // be `stagePos.x % (stageScale * 8)`.
          const scaledOffsetX = stagePos.x % stageScale;
          const scaledOffsetY = stagePos.y % stageScale;

          // Unscale the target position and add the offset to get the absolute position for this anchor.
          const scaledTargetX = targetX * stageScale + scaledOffsetX;
          const scaledTargetY = targetY * stageScale + scaledOffsetY;

          this.log.trace(
            {
              oldPos,
              newPos,
              stageScale,
              stagePos,
              targetX,
              targetY,
              scaledOffsetX,
              scaledOffsetY,
              scaledTargetX,
              scaledTargetY,
            },
            'Anchor drag bound'
          );

          return { x: scaledTargetX, y: scaledTargetY };
        },
        boundBoxFunc: (oldBoundBox, newBoundBox) => {
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
        },
      }),
      proxyRect: new Konva.Rect({
        name: CanvasTransformer.KONVA_PROXY_RECT_NAME,
        listening: false,
        draggable: true,
      }),
    };

    this.konva.transformer.on('transformstart', () => {
      // Just logging in this callback. Called on mouse down of a transform anchor.
      this.log.trace(
        {
          x: this.konva.proxyRect.x(),
          y: this.konva.proxyRect.y(),
          scaleX: this.konva.proxyRect.scaleX(),
          scaleY: this.konva.proxyRect.scaleY(),
          rotation: this.konva.proxyRect.rotation(),
        },
        'Transform started'
      );
    });

    this.konva.transformer.on('transform', () => {
      // This is called when a transform anchor is dragged. By this time, the transform constraints in the above
      // callbacks have been enforced, and the transformer has updated its nodes' attributes. We need to pass the
      // updated attributes to the object group, propagating the transformation on down.
      this.parent.renderer.konva.objectGroup.setAttrs({
        x: this.konva.proxyRect.x(),
        y: this.konva.proxyRect.y(),
        scaleX: this.konva.proxyRect.scaleX(),
        scaleY: this.konva.proxyRect.scaleY(),
        rotation: this.konva.proxyRect.rotation(),
      });
    });

    this.konva.transformer.on('transformend', () => {
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
      this.parent.renderer.konva.objectGroup.setAttrs({
        x: snappedX,
        y: snappedY,
        scaleX: snappedScaleX,
        scaleY: snappedScaleY,
      });

      // Rotation is only retrieved for logging purposes.
      const rotation = this.konva.proxyRect.rotation();

      this.log.trace(
        {
          x,
          y,
          width,
          height,
          scaleX,
          scaleY,
          rotation,
          snappedX,
          snappedY,
          targetWidth,
          targetHeight,
          snappedScaleX,
          snappedScaleY,
        },
        'Transform ended'
      );
    });

    this.konva.proxyRect.on('dragmove', () => {
      // Snap the interaction rect to the nearest pixel
      this.konva.proxyRect.x(Math.round(this.konva.proxyRect.x()));
      this.konva.proxyRect.y(Math.round(this.konva.proxyRect.y()));

      // The bbox should be updated to reflect the new position of the interaction rect, taking into account its padding
      // and border
      this.konva.outlineRect.setAttrs({
        x: this.konva.proxyRect.x() - this.manager.getScaledBboxPadding(),
        y: this.konva.proxyRect.y() - this.manager.getScaledBboxPadding(),
      });

      // The object group is translated by the difference between the interaction rect's new and old positions (which is
      // stored as this.pixelRect)
      this.parent.renderer.konva.objectGroup.setAttrs({
        x: this.konva.proxyRect.x(),
        y: this.konva.proxyRect.y(),
      });
    });
    this.konva.proxyRect.on('dragend', () => {
      if (this.isTransforming) {
        // If we are transforming the entity, we should not push the new position to the state. This will trigger a
        // re-render of the entity and bork the transformation.
        return;
      }

      const position = {
        x: this.konva.proxyRect.x() - this.pixelRect.x,
        y: this.konva.proxyRect.y() - this.pixelRect.y,
      };

      this.log.trace({ position }, 'Position changed');
      this.manager.stateApi.setEntityPosition({ id: this.parent.id, position }, this.parent.type);
    });

    this.subscriptions.add(
      // When the stage scale changes, we may need to re-scale some of the transformer's components. For example,
      // the bbox outline should always be 1 screen pixel wide, so we need to update its stroke width.
      this.manager.stateApi.$stageAttrs.listen((newVal, oldVal) => {
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
    this.subscriptions.add(
      this.manager.toolState.subscribe((newVal, oldVal) => {
        if (newVal.selected !== oldVal.selected) {
          this.syncInteractionState();
        }
      })
    );

    // When the selected entity changes, we need to update the transformer's interaction state.
    this.subscriptions.add(
      this.manager.selectedEntityIdentifier.subscribe(() => {
        this.syncInteractionState();
      })
    );

    this.parent.konva.layer.add(this.konva.outlineRect);
    this.parent.konva.layer.add(this.konva.proxyRect);
    this.parent.konva.layer.add(this.konva.transformer);
  }

  /**
   * Updates the transformer's visual components to match the parent entity's position and bounding box.
   * @param position The position of the parent entity
   * @param bbox The bounding box of the parent entity
   */
  update = (position: Coordinate, bbox: Rect) => {
    const onePixel = this.manager.getScaledPixel();
    const bboxPadding = this.manager.getScaledBboxPadding();

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

    if (this.isPendingRectCalculation || this.pixelRect.width === 0 || this.pixelRect.height === 0) {
      // If the rect is being calculated, or if the rect has no width or height, we can't interact with the transformer
      this.parent.konva.layer.listening(false);
      this.setInteractionMode('off');
      return;
    }

    const toolState = this.manager.stateApi.getToolState();
    const isSelected = this.manager.stateApi.getIsSelected(this.parent.id);

    if (!this.parent.renderer.hasObjects()) {
      // The layer is totally empty, we can just disable the layer
      this.parent.konva.layer.listening(false);
      this.setInteractionMode('off');
      return;
    }

    if (isSelected && !this.isTransforming && toolState.selected === 'move') {
      // We are moving this layer, it must be listening
      this.parent.konva.layer.listening(true);
      this.setInteractionMode('drag');
    } else if (isSelected && this.isTransforming) {
      // When transforming, we want the stage to still be movable if the view tool is selected. If the transformer is
      // active, it will interrupt the stage drag events. So we should disable listening when the view tool is selected.
      if (toolState.selected !== 'view') {
        this.parent.konva.layer.listening(true);
        this.setInteractionMode('all');
      } else {
        this.parent.konva.layer.listening(false);
        this.setInteractionMode('off');
      }
    } else {
      // The layer is not selected, or we are using a tool that doesn't need the layer to be listening - disable interaction stuff
      this.parent.konva.layer.listening(false);
      this.setInteractionMode('off');
    }
  };

  /**
   * Updates the transformer's scale. This is called when the stage is scaled.
   */
  syncScale = () => {
    const onePixel = this.manager.getScaledPixel();
    const bboxPadding = this.manager.getScaledBboxPadding();

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
   */
  startTransform = () => {
    this.log.debug('Starting transform');
    this.isTransforming = true;

    // When transforming, we want the stage to still be movable if the view tool is selected. If the transformer or
    // interaction rect are listening, it will interrupt the stage's drag events. So we should disable listening
    // when the view tool is selected
    const shouldListen = this.manager.stateApi.getToolState().selected !== 'view';
    this.parent.konva.layer.listening(shouldListen);
    this.setInteractionMode('all');
  };

  /**
   * Applies the transformation of the entity.
   */
  applyTransform = async () => {
    this.log.debug('Applying transform');
    await this.parent.renderer.rasterize();
    this.requestRectCalculation();
    this.stopTransform();
  };

  /**
   * Stops the transformation of the entity. If the transformation is in progress, the entity will be reset to its
   * original state.
   */
  stopTransform = () => {
    this.log.debug('Stopping transform');

    this.isTransforming = false;
    this.setInteractionMode('off');

    // Reset the scale of the the entity. We've either replaced the transformed objects with a rasterized image, or
    // canceled a transformation. In either case, the scale should be reset.
    this.resetScale();

    this.updatePosition();
    this.updateBbox();
    this.syncInteractionState();
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

    this.parent.renderer.konva.objectGroup.setAttrs({
      x: position.x + this.pixelRect.x,
      y: position.y + this.pixelRect.y,
      offsetX: this.pixelRect.x,
      offsetY: this.pixelRect.y,
    });

    this.update(position, this.pixelRect);
  };

  /**
   * Sets the transformer to a specific interaction mode.
   * @param interactionMode The mode to set the transformer to. The transformer can be in one of three modes:
   * - 'all': The entity can be moved, resized, and rotated.
   * - 'drag': The entity can be moved.
   * - 'off': The transformer is not interactable.
   */
  setInteractionMode = (interactionMode: 'all' | 'drag' | 'off') => {
    this.interactionMode = interactionMode;
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
    this.log.trace('Updating bbox');

    if (this.isPendingRectCalculation) {
      this.syncInteractionState();
      return;
    }

    // If the bbox has no width or height, that means the layer is fully transparent. This can happen if it is only
    // eraser lines, fully clipped brush lines or if it has been fully erased.
    if (this.pixelRect.width === 0 || this.pixelRect.height === 0) {
      // We shouldn't reset on the first render - the bbox will be calculated on the next render
      if (!this.parent.renderer.hasObjects()) {
        // The layer is fully transparent but has objects - reset it
        this.manager.stateApi.resetEntity({ id: this.parent.id }, this.parent.type);
      }
      this.syncInteractionState();
      return;
    }

    this.syncInteractionState();
    this.update(this.parent.state.position, this.pixelRect);
    this.parent.renderer.konva.objectGroup.setAttrs({
      x: this.parent.state.position.x + this.pixelRect.x,
      y: this.parent.state.position.y + this.pixelRect.y,
      offsetX: this.pixelRect.x,
      offsetY: this.pixelRect.y,
    });
  };

  calculateRect = debounce(() => {
    this.log.debug('Calculating bbox');

    this.isPendingRectCalculation = true;

    if (!this.parent.renderer.hasObjects()) {
      this.log.trace('No objects, resetting bbox');
      this.nodeRect = getEmptyRect();
      this.pixelRect = getEmptyRect();
      this.isPendingRectCalculation = false;
      this.updateBbox();
      return;
    }

    const rect = this.parent.renderer.konva.objectGroup.getClientRect({ skipTransform: true });

    if (!this.parent.renderer.needsPixelBbox()) {
      this.nodeRect = { ...rect };
      this.pixelRect = { ...rect };
      this.isPendingRectCalculation = false;
      this.log.trace({ nodeRect: this.nodeRect, pixelRect: this.pixelRect }, 'Got bbox from client rect');
      this.updateBbox();
      return;
    }

    // We have eraser strokes - we must calculate the bbox using pixel data

    const clone = this.parent.renderer.konva.objectGroup.clone();
    const canvas = clone.toCanvas();
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    const imageData = ctx.getImageData(0, 0, rect.width, rect.height);
    this.manager.requestBbox(
      { buffer: imageData.data.buffer, width: imageData.width, height: imageData.height },
      (extents) => {
        if (extents) {
          const { minX, minY, maxX, maxY } = extents;
          this.nodeRect = { ...rect };
          this.pixelRect = {
            x: rect.x + minX,
            y: rect.y + minY,
            width: maxX - minX,
            height: maxY - minY,
          };
        } else {
          this.nodeRect = getEmptyRect();
          this.pixelRect = getEmptyRect();
        }
        this.isPendingRectCalculation = false;
        this.log.trace({ nodeRect: this.nodeRect, pixelRect: this.pixelRect, extents }, `Got bbox from worker`);
        this.updateBbox();
        clone.destroy();
      }
    );
  }, CanvasManager.BBOX_DEBOUNCE_MS);

  requestRectCalculation = () => {
    this.isPendingRectCalculation = true;
    this.syncInteractionState();
    this.calculateRect();
  };

  _enableTransform = () => {
    this.isTransformEnabled = true;
    this.konva.transformer.visible(true);
    this.konva.transformer.listening(true);
    this.konva.transformer.nodes([this.konva.proxyRect]);
  };

  _disableTransform = () => {
    this.isTransformEnabled = false;
    this.konva.transformer.visible(false);
    this.konva.transformer.listening(false);
    this.konva.transformer.nodes([]);
  };

  _enableDrag = () => {
    this.isDragEnabled = true;
    this.konva.proxyRect.visible(true);
    this.konva.proxyRect.listening(true);
  };

  _disableDrag = () => {
    this.isDragEnabled = false;
    this.konva.proxyRect.visible(false);
    this.konva.proxyRect.listening(false);
  };

  _showBboxOutline = () => {
    this.konva.outlineRect.visible(true);
  };

  _hideBboxOutline = () => {
    this.konva.outlineRect.visible(false);
  };

  /**
   * Gets a JSON-serializable object that describes the transformer.
   */
  repr = () => {
    return {
      id: this.id,
      type: CanvasTransformer.TYPE,
      mode: this.interactionMode,
      isTransformEnabled: this.isTransformEnabled,
      isDragEnabled: this.isDragEnabled,
    };
  };

  /**
   * Destroys the transformer, cleaning up any subscriptions.
   */
  destroy = () => {
    this.log.trace('Destroying transformer');
    for (const cleanup of this.subscriptions) {
      this.log.trace('Cleaning up listener');
      cleanup();
    }
    this.konva.outlineRect.destroy();
    this.konva.transformer.destroy();
    this.konva.proxyRect.destroy();
  };
}
