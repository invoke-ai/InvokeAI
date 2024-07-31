import type { JSONObject } from 'common/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { Logger } from 'roarr';

export abstract class CanvasEntity {
  id: string;
  _manager: CanvasManager;
  _log: Logger;

  constructor(id: string, manager: CanvasManager) {
    this.id = id;
    this._manager = manager;
    this._log = this._manager.buildLogger(this._getLoggingContext);
  }
  /**
   * Get a serializable representation of the entity.
   */
  abstract repr(): JSONObject;

  _getLoggingContext = (extra?: Record<string, unknown>) => {
    return {
      ...this._manager._getLoggingContext(),
      layerId: this.id,
      ...extra,
    };
  };
}
