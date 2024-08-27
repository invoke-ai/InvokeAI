import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleABC } from 'features/controlLayers/konva/CanvasModuleABC';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { GenerationMode } from 'features/controlLayers/store/types';
import { LRUCache } from 'lru-cache';
import type { Logger } from 'roarr';

export class CanvasCacheModule extends CanvasModuleABC {
  readonly type = 'cache';

  id: string;
  path: string[];
  log: Logger;
  manager: CanvasManager;
  subscriptions = new Set<() => void>();

  imageNameCache = new LRUCache<string, string>({ max: 100 });
  canvasElementCache = new LRUCache<string, HTMLCanvasElement>({ max: 32 });
  generationModeCache = new LRUCache<string, GenerationMode>({ max: 100 });

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('cache');
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.debug('Creating cache module');
  }

  clearAll = () => {
    this.canvasElementCache.clear();
    this.imageNameCache.clear();
    this.generationModeCache.clear();
  };

  repr = () => {
    return {
      id: this.id,
      path: this.path,
      type: this.type,
    };
  };

  destroy = () => {
    this.log.debug('Destroying cache module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.clearAll();
  };

  getLoggingContext = () => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
