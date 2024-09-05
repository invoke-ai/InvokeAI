import type { SerializableObject } from 'common/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { Logger } from 'roarr';

/**
 * Base class for all canvas modules.
 */
export abstract class CanvasModuleBase {
  /**
   * The type of the module.
   */
  abstract readonly type: string;
  /**
   * The unique identifier of the module.
   *
   * If the module is associated with an entity, this should be the entity's id. Otherwise, the id should be based on
   * the module's type. The `getPrefixedId` utility should be used for generating ids.
   *
   * @example
   * ```ts
   * this.id = getPrefixedId(this.type);
   * // this.id -> "raster_layer:aS2NREsrlz"
   * ```
   */
  abstract readonly id: string;
  /**
   * The path of the module in the canvas module tree.
   *
   * Modules should use the manager's `buildPath` method to set this value.
   *
   * @example
   * ```ts
   * this.path = this.manager.buildPath(this);
   * // this.path -> ["manager:3PWJWmHbou", "raster_layer:aS2NREsrlz", "entity_renderer:sfLO4j1B0n", "brush_line:Zrsu8gpZMd"]
   * ```
   */
  abstract readonly path: string[];
  /**
   * The parent module. This may be the canvas manager or another module.
   */
  abstract readonly parent: CanvasModuleBase;
  /**
   * The canvas manager.
   */
  abstract readonly manager: CanvasManager;
  /**
   * The logger for the module. The logger must be a `ROARR` logger.
   *
   * Modules should use the manager's `buildLogger` method to set this value.
   *
   * @example
   * ```ts
   * this.log = this.manager.buildLogger(this);
   * ```
   */
  abstract readonly log: Logger;

  /**
   * An optional method that initializes the module. This method is called after all modules have been created.
   *
   * Use this method to perform any setup that requires all modules to be created. For example, setting some initial
   * state or doing an initial render.
   */
  initialize?: () => void = undefined;

  /**
   * Returns a logging context object that includes relevant information about the module.
   * Canvas modules may override this method to include additional information in the logging context, but should
   * always include the parent's logging context.
   *
   * The default implementation includes the parent context and the module's path as a string.
   *
   * @example
   * ```ts
   * getLoggingContext = () => {
   *   return {
   *     ...this.parent.getLoggingContext(),
   *     path: this.path.join(' > '),
   *     someImportantValue: this.someImportantValue,
   *   };
   * };
   * ```
   */
  getLoggingContext: () => SerializableObject = () => {
    return {
      ...this.parent.getLoggingContext(),
      path: this.path.join(' > '),
    };
  };

  /**
   * Cleans up the module when it is disposed.
   *
   * Canvas modules may override this method to clean up any loose ends. For example:
   * - Destroy Konva nodes
   * - Unsubscribe from any subscriptions
   * - Abort async operations
   * - Close websockets
   * - Terminate workers
   *
   * This method is called when the module is disposed. For example:
   * - When an entity is deleted and its module is destroyed
   * - When the canvas manager is destroyed
   *
   * The default implementation only logs a message.
   *
   * @example
   * ```ts
   * destroy = () => {
   *  this.log('Destroying module');
   *  this.subscriptions.forEach((unsubscribe) => unsubscribe());
   *  this.subscriptions.clear();
   *  this.konva.group.destroy();
   * };
   * ```
   */
  destroy: () => void = () => {
    this.log('Destroying module');
  };

  /**
   * Returns a serializable representation of the module.
   * Canvas modules may override this method to include additional information in the representation.
   * The default implementation includes id, type, and path.
   *
   * @example
   * ```ts
   * repr = () => {
   *   return {
   *     id: this.id,
   *     type: this.type,
   *     path: this.path,
   *     state: deepClone(this.state),
   *   };
   * };
   * ```
   */
  repr: () => SerializableObject = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
    };
  };
}
