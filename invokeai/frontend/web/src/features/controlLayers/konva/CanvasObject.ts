import type { JSONObject } from 'common/types';
import type { CanvasControlAdapter } from 'features/controlLayers/konva/CanvasControlAdapter';
import type { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasStagingArea } from 'features/controlLayers/konva/CanvasStagingArea';
import type { Logger } from 'roarr';

export abstract class CanvasObject {
  id: string;

  parent: CanvasLayer | CanvasStagingArea | CanvasControlAdapter;
  manager: CanvasManager;
  log: Logger;

  constructor(id: string, parent: CanvasLayer | CanvasStagingArea | CanvasControlAdapter) {
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.log = this.manager.buildLogger(this.getLoggingContext);
  }

  /**
   * Get a serializable representation of the object.
   */
  abstract repr(): JSONObject;

  /**
   * Get the logging context for this object.
   * @param extra Extra data to merge into the context
   * @returns The logging context for this object
   */
  getLoggingContext = (extra?: Record<string, unknown>) => {
    return {
      ...this.parent.getLoggingContext(),
      objectId: this.id,
      ...extra,
    };
  };
}
