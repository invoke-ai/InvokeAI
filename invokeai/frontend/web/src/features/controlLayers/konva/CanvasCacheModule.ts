import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { GenerationMode } from 'features/controlLayers/store/types';
import { LRUCache } from 'lru-cache';
import type { Logger } from 'roarr';

type CanvasCacheModuleConfig = {
  /**
   * The maximum size of the image name cache.
   */
  imageNameCacheSize: number;
  /**
   * The maximum size of the canvas element cache.
   */
  canvasElementCacheSize: number;
  /**
   * The maximum size of the generation mode cache.
   */
  generationModeCacheSize: number;
};

const DEFAULT_CONFIG: CanvasCacheModuleConfig = {
  imageNameCacheSize: 100,
  canvasElementCacheSize: 32,
  generationModeCacheSize: 100,
};

/**
 * A cache module for storing the results of expensive calculations. For example, when we rasterize a layer and upload
 * it to the server, we store the resultant image name in this cache for future use.
 */
export class CanvasCacheModule extends CanvasModuleBase {
  readonly type = 'cache';
  readonly id: string;
  readonly path: string[];
  readonly log: Logger;
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;

  config: CanvasCacheModuleConfig = DEFAULT_CONFIG;

  /**
   * A cache for storing image names. Used as a cache for results of layer/canvas/entity exports. For example, when we
   * rasterize a layer and upload it to the server, we store the image name in this cache.
   *
   * The cache key is a hash of the exported entity's state and the export rect.
   */
  imageNameCache = new LRUCache<string, string>({ max: this.config.imageNameCacheSize });

  /**
   * A cache for storing canvas elements. Similar to the image name cache, but for canvas elements. The primary use is
   * for caching composite layers. For example, the canvas compositor module uses this to store the canvas elements for
   * individual raster layers when creating a composite of the layers.
   *
   * The cache key is a hash of the exported entity's state and the export rect.
   */
  canvasElementCache = new LRUCache<string, HTMLCanvasElement>({ max: this.config.canvasElementCacheSize });
  /**
   * A cache for the generation mode calculation, which is fairly expensive.
   *
   * The cache key is a hash of all the objects that contribute to the generation mode calculation (e.g. the composite
   * raster layer, the composite inpaint mask, and bounding box), and the value is the generation mode.
   */
  generationModeCache = new LRUCache<string, GenerationMode>({ max: this.config.generationModeCacheSize });

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('cache');
    this.manager = manager;
    this.parent = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating cache module');
  }

  /**
   * Clears all caches.
   */
  clearAll = () => {
    this.canvasElementCache.clear();
    this.imageNameCache.clear();
    this.generationModeCache.clear();
  };

  destroy = () => {
    this.log.debug('Destroying cache module');
    this.clearAll();
  };
}
