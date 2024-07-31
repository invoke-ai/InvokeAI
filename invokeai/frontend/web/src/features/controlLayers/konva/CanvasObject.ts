import type { JSONObject } from 'common/types';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasStagingArea } from 'features/controlLayers/konva/CanvasStagingArea';
import type { Logger } from 'roarr';

export abstract class CanvasObject {
  id: string;

  _parent: CanvasLayer | CanvasStagingArea;
  _manager: CanvasManager;
  _log: Logger;

  constructor(id: string, parent: CanvasLayer | CanvasStagingArea) {
    this.id = id;
    this._parent = parent;
    this._manager = parent._manager;
    this._log = this._manager.buildLogger(this._getLoggingContext);
  }

  /**
   * Destroy the object's konva nodes.
   */
  abstract destroy(): void;

  /**
   * Set the visibility of the object's konva nodes.
   */
  abstract setVisibility(isVisible: boolean): void;

  /**
   * Get a serializable representation of the object.
   */
  abstract repr(): JSONObject;

  /**
   * Get the logging context for this object.
   * @param extra Extra data to merge into the context
   * @returns The logging context for this object
   */
  _getLoggingContext = (extra?: Record<string, unknown>) => {
    return {
      ...this._parent._getLoggingContext(),
      objectId: this.id,
      ...extra,
    };
  };
}
