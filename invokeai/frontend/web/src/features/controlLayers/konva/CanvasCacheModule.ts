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

export class CanvasCacheModule extends CanvasModuleBase {
  readonly type = 'cache';

  id: string;
  path: string[];
  log: Logger;
  parent: CanvasManager;
  manager: CanvasManager;

  config: CanvasCacheModuleConfig = DEFAULT_CONFIG;

  imageNameCache = new LRUCache<string, string>({ max: this.config.imageNameCacheSize });
  canvasElementCache = new LRUCache<string, HTMLCanvasElement>({ max: this.config.canvasElementCacheSize });
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
