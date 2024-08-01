import type { JSONObject } from 'common/types';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasInteractionRect {
  static TYPE = 'interaction_rect';

  id: string;
  parent: CanvasLayer;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: (extra?: JSONObject) => JSONObject;

  konva: {
    rect: Konva.Rect;
  };

  constructor(parent: CanvasLayer) {
    this.id = getPrefixedId(CanvasInteractionRect.TYPE);
    this.parent = parent;
    this.manager = parent.manager;

    this.getLoggingContext = this.manager.buildObjectGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.konva = {
      rect: new Konva.Rect({
        name: CanvasInteractionRect.TYPE,
        listening: false,
        draggable: true,
        // fill: 'rgba(255,0,0,0.5)',
      }),
    };

    this.konva.rect.on('dragmove', () => {
      // Snap the interaction rect to the nearest pixel
      this.konva.rect.x(Math.round(this.konva.rect.x()));
      this.konva.rect.y(Math.round(this.konva.rect.y()));

      // The bbox should be updated to reflect the new position of the interaction rect, taking into account its padding
      // and border
      this.parent.konva.bbox.setAttrs({
        x: this.konva.rect.x() - this.manager.getScaledBboxPadding(),
        y: this.konva.rect.y() - this.manager.getScaledBboxPadding(),
      });

      // The object group is translated by the difference between the interaction rect's new and old positions (which is
      // stored as this.bbox)
      this.parent.konva.objectGroup.setAttrs({
        x: this.konva.rect.x(),
        y: this.konva.rect.y(),
      });
    });
    this.konva.rect.on('dragend', () => {
      if (this.parent.isTransforming) {
        // When the user cancels the transformation, we need to reset the layer, so we should not update the layer's
        // positition while we are transforming - bail out early.
        return;
      }

      const position = {
        x: this.konva.rect.x() - this.parent.bbox.x,
        y: this.konva.rect.y() - this.parent.bbox.y,
      };

      this.log.trace({ position }, 'Position changed');
      this.manager.stateApi.onPosChanged({ id: this.id, position }, 'layer');
    });
  }

  repr = () => {
    return {
      id: this.id,
      type: CanvasInteractionRect.TYPE,
    };
  };
}
