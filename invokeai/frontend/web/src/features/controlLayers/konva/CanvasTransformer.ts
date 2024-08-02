import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { Subscription } from 'features/controlLayers/konva/util';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Coordinate, GetLoggingContext, Rect } from 'features/controlLayers/store/types';
import Konva from 'konva';
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
  static TRANSFORMER_NAME = `${CanvasTransformer.TYPE}:transformer`;
  static PROXY_RECT_NAME = `${CanvasTransformer.TYPE}:proxy_rect`;
  static BBOX_OUTLINE_NAME = `${CanvasTransformer.TYPE}:bbox_outline`;
  static STROKE_COLOR = 'hsl(200deg 76% 59%)'; // `invokeBlue.400

  id: string;
  parent: CanvasLayer;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;
  subscriptions: Subscription[];

  /**
   * The current mode of the transformer:
   * - 'transform': The entity can be moved, resized, and rotated
   * - 'drag': The entity can only be moved
   * - 'off': The transformer is disabled
   */
  mode: 'transform' | 'drag' | 'off';

  /**
   * Whether dragging is enabled. Dragging is enabled in both 'transform' and 'drag' modes.
   */
  isDragEnabled: boolean;

  /**
   * Whether transforming is enabled. Transforming is enabled only in 'transform' mode.
   */
  isTransformEnabled: boolean;

  konva: {
    transformer: Konva.Transformer;
    proxyRect: Konva.Rect;
    bboxOutline: Konva.Rect;
  };

  constructor(parent: CanvasLayer) {
    this.id = getPrefixedId(CanvasTransformer.TYPE);
    this.parent = parent;
    this.manager = parent.manager;

    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.subscriptions = [];

    this.mode = 'off';
    this.isDragEnabled = false;
    this.isTransformEnabled = false;

    this.konva = {
      bboxOutline: new Konva.Rect({
        listening: false,
        draggable: false,
        name: CanvasTransformer.BBOX_OUTLINE_NAME,
        stroke: CanvasTransformer.STROKE_COLOR,
        perfectDrawEnabled: false,
        strokeHitEnabled: false,
      }),
      transformer: new Konva.Transformer({
        name: CanvasTransformer.TRANSFORMER_NAME,
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
          if (this.manager.stateApi.getShiftKey()) {
            if (Math.abs(newBoundBox.rotation % (Math.PI / 4)) > 0) {
              return oldBoundBox;
            }
          }

          return newBoundBox;
        },
      }),
      proxyRect: new Konva.Rect({
        name: CanvasTransformer.PROXY_RECT_NAME,
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
      this.parent.konva.objectGroup.setAttrs({
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
      this.parent.konva.objectGroup.setAttrs({
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
      this.konva.bboxOutline.setAttrs({
        x: this.konva.proxyRect.x() - this.manager.getScaledBboxPadding(),
        y: this.konva.proxyRect.y() - this.manager.getScaledBboxPadding(),
      });

      // The object group is translated by the difference between the interaction rect's new and old positions (which is
      // stored as this.bbox)
      this.parent.konva.objectGroup.setAttrs({
        x: this.konva.proxyRect.x(),
        y: this.konva.proxyRect.y(),
      });
    });
    this.konva.proxyRect.on('dragend', () => {
      if (this.parent.isTransforming) {
        // When the user cancels the transformation, we need to reset the layer, so we should not update the layer's
        // positition while we are transforming - bail out early.
        return;
      }

      const position = {
        x: this.konva.proxyRect.x() - this.parent.bbox.x,
        y: this.konva.proxyRect.y() - this.parent.bbox.y,
      };

      this.log.trace({ position }, 'Position changed');
      this.manager.stateApi.onPosChanged({ id: this.parent.id, position }, 'layer');
    });

    this.subscriptions.push(
      // When the stage scale changes, we may need to re-scale some of the transformer's components. For example,
      // the bbox outline should always be 1 screen pixel wide, so we need to update its stroke width.
      this.manager.stateApi.onStageAttrsChanged((newAttrs, oldAttrs) => {
        if (newAttrs.scale !== oldAttrs?.scale) {
          this.scale();
        }
      })
    );

    this.subscriptions.push(
      // While the user holds shift, we want to snap rotation to 45 degree increments. Listen for the shift key state
      // and update the snap angles accordingly.
      this.manager.stateApi.onShiftChanged((isPressed) => {
        this.konva.transformer.rotationSnaps(isPressed ? [0, 45, 90, 135, 180, 225, 270, 315] : []);
      })
    );
  }

  /**
   * Updates the transformer's visual components to match the parent entity's position and bounding box.
   * @param position The position of the parent entity
   * @param bbox The bounding box of the parent entity
   */
  update = (position: Coordinate, bbox: Rect) => {
    const onePixel = this.manager.getScaledPixel();
    const bboxPadding = this.manager.getScaledBboxPadding();

    this.konva.bboxOutline.setAttrs({
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
   * Updates the transformer's scale. This is called when the stage is scaled.
   */
  scale = () => {
    const onePixel = this.manager.getScaledPixel();
    const bboxPadding = this.manager.getScaledBboxPadding();

    this.konva.bboxOutline.setAttrs({
      x: this.konva.proxyRect.x() - bboxPadding,
      y: this.konva.proxyRect.y() - bboxPadding,
      width: this.konva.proxyRect.width() * this.konva.proxyRect.scaleX() + bboxPadding * 2,
      height: this.konva.proxyRect.height() * this.konva.proxyRect.scaleY() + bboxPadding * 2,
      strokeWidth: onePixel,
    });
    this.konva.transformer.forceUpdate();
  };

  /**
   * Sets the transformer to a specific mode.
   * @param mode The mode to set the transformer to. The transformer can be in one of three modes:
   * - 'transform': The entity can be moved, resized, and rotated
   * - 'drag': The entity can only be moved
   * - 'off': The transformer is disabled
   */
  setMode = (mode: 'transform' | 'drag' | 'off') => {
    this.mode = mode;
    if (mode === 'drag') {
      this._enableDrag();
      this._disableTransform();
      this._showBboxOutline();
    } else if (mode === 'transform') {
      this._enableDrag();
      this._enableTransform();
      this._hideBboxOutline();
    } else if (mode === 'off') {
      this._disableDrag();
      this._disableTransform();
      this._hideBboxOutline();
    }
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
    this.konva.bboxOutline.visible(true);
  };

  _hideBboxOutline = () => {
    this.konva.bboxOutline.visible(false);
  };

  getNodes = () => [this.konva.transformer, this.konva.proxyRect, this.konva.bboxOutline];

  repr = () => {
    return {
      id: this.id,
      type: CanvasTransformer.TYPE,
      mode: this.mode,
      isTransformEnabled: this.isTransformEnabled,
      isDragEnabled: this.isDragEnabled,
    };
  };

  destroy = () => {
    this.log.trace('Destroying transformer');
    for (const { name, unsubscribe } of this.subscriptions) {
      this.log.trace({ name }, 'Cleaning up listener');
      unsubscribe();
    }
    this.konva.bboxOutline.destroy();
    this.konva.transformer.destroy();
    this.konva.proxyRect.destroy();
  };
}
