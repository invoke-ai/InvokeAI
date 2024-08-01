import type { JSONObject } from 'common/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { Logger } from 'roarr';

export abstract class CanvasEntity {
  id: string;
  manager: CanvasManager;
  log: Logger;

  constructor(id: string, manager: CanvasManager) {
    this.id = id;
    this.manager = manager;
    this.log = this.manager.buildLogger(this.getLoggingContext);
  }
  /**
   * Get a serializable representation of the entity.
   */
  abstract repr(): JSONObject;

  getLoggingContext = (extra?: Record<string, unknown>) => {
    return {
      ...this.manager.getLoggingContext(),
      layerId: this.id,
      ...extra,
    };
  };
}
