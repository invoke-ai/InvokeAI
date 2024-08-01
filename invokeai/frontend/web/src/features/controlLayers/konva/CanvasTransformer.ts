import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import { CanvasObject } from 'features/controlLayers/konva/CanvasObject';
import { nanoid } from 'features/controlLayers/konva/util';
import type { Coordinate } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasTransformer extends CanvasObject {
  static TYPE = 'transformer';

  isActive: boolean;
  konva: {
    transformer: Konva.Transformer;
  };

  constructor(parent: CanvasLayer) {
    super(`${CanvasTransformer.TYPE}:${nanoid()}`, parent);

    this.isActive = false;
    this.konva = {
      transformer: new Konva.Transformer({
        name: CanvasTransformer.TYPE,
        // The transformer will use the interaction rect as a proxy for the entity it is transforming.
        nodes: [parent.konva.interactionRect],
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
        stroke: 'hsl(200deg 76% 59%)',
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
    };

    this.konva.transformer.on('transformstart', () => {
      // Just logging in this callback. Called on mouse down of a transform anchor.
      this.log.trace(
        {
          x: parent.konva.interactionRect.x(),
          y: parent.konva.interactionRect.y(),
          scaleX: parent.konva.interactionRect.scaleX(),
          scaleY: parent.konva.interactionRect.scaleY(),
          rotation: parent.konva.interactionRect.rotation(),
        },
        'Transform started'
      );
    });

    this.konva.transformer.on('transform', () => {
      // This is called when a transform anchor is dragged. By this time, the transform constraints in the above
      // callbacks have been enforced, and the transformer has updated its nodes' attributes. We need to pass the
      // updated attributes to the object group, propagating the transformation on down.
      parent.konva.objectGroup.setAttrs({
        x: parent.konva.interactionRect.x(),
        y: parent.konva.interactionRect.y(),
        scaleX: parent.konva.interactionRect.scaleX(),
        scaleY: parent.konva.interactionRect.scaleY(),
        rotation: parent.konva.interactionRect.rotation(),
      });
    });

    this.konva.transformer.on('transformend', () => {
      // Called on mouse up on an anchor. We'll do some final snapping to ensure the transformer is pixel-perfect.

      // Snap the position to the nearest pixel.
      const x = parent.konva.interactionRect.x();
      const y = parent.konva.interactionRect.y();
      const snappedX = Math.round(x);
      const snappedY = Math.round(y);

      // The transformer doesn't modify the width and height. It only modifies scale. We'll need to apply the scale to
      // the width and height, round them to the nearest pixel, and finally calculate a new scale that will result in
      // the snapped width and height.
      const width = parent.konva.interactionRect.width();
      const height = parent.konva.interactionRect.height();
      const scaleX = parent.konva.interactionRect.scaleX();
      const scaleY = parent.konva.interactionRect.scaleY();

      // Determine the target width and height, rounded to the nearest pixel. Must be >= 1. Because the scales can be
      // negative, we need to take the absolute value of the width and height.
      const targetWidth = Math.max(Math.abs(Math.round(width * scaleX)), 1);
      const targetHeight = Math.max(Math.abs(Math.round(height * scaleY)), 1);

      // Calculate the scale we need to use to get the target width and height. Restore the sign of the scales.
      const snappedScaleX = (targetWidth / width) * Math.sign(scaleX);
      const snappedScaleY = (targetHeight / height) * Math.sign(scaleY);

      // Update interaction rect and object group attributes.
      parent.konva.interactionRect.setAttrs({
        x: snappedX,
        y: snappedY,
        scaleX: snappedScaleX,
        scaleY: snappedScaleY,
      });
      parent.konva.objectGroup.setAttrs({
        x: snappedX,
        y: snappedY,
        scaleX: snappedScaleX,
        scaleY: snappedScaleY,
      });

      // Rotation is only retrieved for logging purposes.
      const rotation = parent.konva.interactionRect.rotation();

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

    this.manager.stateApi.onShiftChanged((isPressed) => {
      // While the user holds shift, we want to snap rotation to 45 degree increments. Listen for the shift key state
      // and update the snap angles accordingly.
      this.konva.transformer.rotationSnaps(isPressed ? [0, 45, 90, 135, 180, 225, 270, 315] : []);
    });
  }

  /**
   * Activate the transformer. This will make it visible and listening for events.
   */
  activate = () => {
    this.isActive = true;
    this.konva.transformer.visible(true);
    this.konva.transformer.listening(true);
  };

  /**
   * Deactivate the transformer. This will make it invisible and not listening for events.
   */
  deactivate = () => {
    this.isActive = false;
    this.konva.transformer.visible(false);
    this.konva.transformer.listening(false);
  };

  repr = () => {
    return {
      id: this.id,
      type: CanvasTransformer.TYPE,
      isActive: this.isActive,
    };
  };
}
