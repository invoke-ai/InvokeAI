import type { SerializableObject } from 'common/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { GenerationMode } from 'features/controlLayers/store/types';
import { LRUCache } from 'lru-cache';
import type { Logger } from 'roarr';

export class CanvasCacheModule {
  id: string;
  path: string[];
  log: Logger;
  manager: CanvasManager;

  imageNameCache = new LRUCache<string, string>({ max: 100 });
  canvasElementCache = new LRUCache<string, HTMLCanvasElement>({ max: 32 });
  generationModeCache = new LRUCache<string, GenerationMode>({ max: 100 });

  constructor(manager: CanvasManager) {
    this.id = getPrefixedId('cache');
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug('Creating canvas cache');
  }

  clearAll = () => {
    this.canvasElementCache.clear();
    this.imageNameCache.clear();
    this.generationModeCache.clear();
  };

  getLoggingContext = (): SerializableObject => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
