import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { Transparency } from 'features/controlLayers/konva/util';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { GenerationMode } from 'features/controlLayers/store/types';
import { LRUCache } from 'lru-cache';
import type { Logger } from 'roarr';

type GetCacheEntryWithFallbackArg<T extends NonNullable<unknown>> = {
  cache: LRUCache<string, T>;
  key: string;
  getValue: () => Promise<T>;
  onHit?: (value: T) => void;
  onMiss?: () => void;
};

type CanvasCacheModuleConfig = {
  /**
   * The maximum size of the image name cache.
   */
  imageNameCacheSize: number;
  /**
   * The maximum size of the image data cache.
   */
  imageDataCacheSize: number;
  /**
   * The maximum size of the transparency calculation cache.
   */
  transparencyCalculationCacheSize: number;
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
  imageNameCacheSize: 1000,
  imageDataCacheSize: 32,
  transparencyCalculationCacheSize: 1000,
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
   * A cache for storing image names.
   *
   * For example, the key might be a hash of a composite of entities with the uploaded image name as the value.
   */
  imageNameCache = new LRUCache<string, string>({ max: this.config.imageNameCacheSize });

  /**
   * A cache for storing canvas elements.
   *
   * For example, the key might be a hash of a composite of entities with the canvas element as the value.
   */
  canvasElementCache = new LRUCache<string, HTMLCanvasElement>({ max: this.config.canvasElementCacheSize });

  /**
   * A cache for image data objects.
   *
   * For example, the key might be a hash of a composite of entities with the image data as the value.
   */
  imageDataCache = new LRUCache<string, ImageData>({ max: this.config.imageDataCacheSize });

  /**
   * A cache for transparency calculation results.
   *
   * For example, the key might be a hash of a composite of entities with the transparency as the value.
   */
  transparencyCalculationCache = new LRUCache<string, Transparency>({ max: this.config.imageDataCacheSize });

  /**
   * A cache for generation mode calculation results.
   *
   * For example, the key might be a hash of a composite of raster and inpaint mask entities with the generation mode
   * as the value.
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
   * A helper function for getting a cache entry with a fallback.
   * @param param0.cache The LRUCache to get the entry from.
   * @param param0.key The key to use to retrieve the entry.
   * @param param0.getValue An async function to generate the value if the entry is not in the cache.
   * @param param0.onHit An optional function to call when the entry is in the cache.
   * @param param0.onMiss An optional function to call when the entry is not in the cache.
   * @returns
   */
  static getWithFallback = async <T extends NonNullable<unknown>>({
    cache,
    getValue,
    key,
    onHit,
    onMiss,
  }: GetCacheEntryWithFallbackArg<T>): Promise<T> => {
    let value = cache.get(key);
    if (value === undefined) {
      onMiss?.();
      value = await getValue();
      cache.set(key, value);
    } else {
      onHit?.(value);
    }
    return value;
  };

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
