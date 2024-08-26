import type { AppSocket } from 'app/hooks/useSocketIO';
import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import type { SerializableObject } from 'common/types';
import { SyncableMap } from 'common/util/SyncableMap/SyncableMap';
import { CanvasCacheModule } from 'features/controlLayers/konva/CanvasCacheModule';
import { CanvasCompositorModule } from 'features/controlLayers/konva/CanvasCompositorModule';
import { CanvasFilterModule } from 'features/controlLayers/konva/CanvasFilterModule';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasRenderingModule } from 'features/controlLayers/konva/CanvasRenderingModule';
import { CanvasStageModule } from 'features/controlLayers/konva/CanvasStageModule';
import { CanvasWorkerModule } from 'features/controlLayers/konva/CanvasWorkerModule.js';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type Konva from 'konva';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';

import { CanvasBackgroundModule } from './CanvasBackgroundModule';
import type { CanvasLayerAdapter } from './CanvasLayerAdapter';
import type { CanvasMaskAdapter } from './CanvasMaskAdapter';
import { CanvasPreviewModule } from './CanvasPreviewModule';
import { CanvasStateApiModule } from './CanvasStateApiModule';

export const $canvasManager = atom<CanvasManager | null>(null);

export class CanvasManager extends CanvasModuleBase {
  readonly type = 'manager';

  id: string;
  path: string[];
  manager: CanvasManager;
  log: Logger;

  store: AppStore;
  socket: AppSocket;

  subscriptions = new Set<() => void>();

  adapters = {
    rasterLayers: new SyncableMap<string, CanvasLayerAdapter>(),
    controlLayers: new SyncableMap<string, CanvasLayerAdapter>(),
    regionMasks: new SyncableMap<string, CanvasMaskAdapter>(),
    inpaintMasks: new SyncableMap<string, CanvasMaskAdapter>(),
    getAll: (): (CanvasLayerAdapter | CanvasMaskAdapter)[] => {
      return [
        ...this.adapters.rasterLayers.values(),
        ...this.adapters.controlLayers.values(),
        ...this.adapters.regionMasks.values(),
        ...this.adapters.inpaintMasks.values(),
      ];
    },
  };

  stateApi: CanvasStateApiModule;
  preview: CanvasPreviewModule;
  background: CanvasBackgroundModule;
  filter: CanvasFilterModule;
  stage: CanvasStageModule;
  worker: CanvasWorkerModule;
  cache: CanvasCacheModule;
  renderer: CanvasRenderingModule;
  compositor: CanvasCompositorModule;

  _isDebugging: boolean = false;

  constructor(stage: Konva.Stage, container: HTMLDivElement, store: AppStore, socket: AppSocket) {
    super();
    this.id = getPrefixedId(this.type);
    this.path = [this.id];
    this.manager = this;
    this.log = logger('canvas').child((message) => {
      return {
        ...message,
        context: {
          ...this.getLoggingContext(),
          ...message.context,
        },
      };
    });
    this.log.debug('Creating canvas manager module');

    this.store = store;
    this.socket = socket;

    this.stateApi = new CanvasStateApiModule(this.store, this);
    this.stage = new CanvasStageModule(stage, container, this);
    this.worker = new CanvasWorkerModule(this);
    this.cache = new CanvasCacheModule(this);
    this.renderer = new CanvasRenderingModule(this);
    this.preview = new CanvasPreviewModule(this);
    this.filter = new CanvasFilterModule(this);

    this.compositor = new CanvasCompositorModule(this);
    this.stage.addLayer(this.preview.getLayer());

    this.background = new CanvasBackgroundModule(this);
    this.stage.addLayer(this.background.konva.layer);
  }

  enableDebugging() {
    this._isDebugging = true;
    this.logDebugInfo();
  }

  disableDebugging() {
    this._isDebugging = false;
  }

  initialize = () => {
    this.log.debug('Initializing canvas manager module');

    // These atoms require the canvas manager to be set up before we can provide their initial values
    this.stateApi.$transformingEntity.set(null);
    this.stateApi.$toolState.set(this.stateApi.getToolState());
    this.stateApi.$selectedEntityIdentifier.set(this.stateApi.getCanvasState().selectedEntityIdentifier);
    this.stateApi.$currentFill.set(this.stateApi.getCurrentFill());
    this.stateApi.$selectedEntity.set(this.stateApi.getSelectedEntity());

    this.subscriptions.add(this.store.subscribe(this.renderer.render));
    this.stage.initialize();
  };

  destroy = () => {
    this.log.debug('Destroying canvas manager module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    for (const adapter of this.adapters.getAll()) {
      adapter.destroy();
    }
    this.stateApi.destroy();
    this.preview.destroy();
    this.background.destroy();
    this.filter.destroy();
    this.worker.destroy();
    this.renderer.destroy();
    this.compositor.destroy();
    this.stage.destroy();
    $canvasManager.set(null);
  };

  setCanvasManager = () => {
    this.log.debug('Setting canvas manager global');
    $canvasManager.set(this);
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      rasterLayers: Array.from(this.adapters.rasterLayers.values()).map((adapter) => adapter.repr()),
      controlLayers: Array.from(this.adapters.controlLayers.values()).map((adapter) => adapter.repr()),
      inpaintMasks: Array.from(this.adapters.inpaintMasks.values()).map((adapter) => adapter.repr()),
      regionMasks: Array.from(this.adapters.regionMasks.values()).map((adapter) => adapter.repr()),
      stateApi: this.stateApi.repr(),
      preview: this.preview.repr(),
      background: this.background.repr(),
      filter: this.filter.repr(),
      worker: this.worker.repr(),
      renderer: this.renderer.repr(),
      compositor: this.compositor.repr(),
      stage: this.stage.repr(),
    };
  };

  getLoggingContext = (): SerializableObject => {
    return {
      path: this.path.join('.'),
    };
  };

  buildLogger = (getContext: () => SerializableObject): Logger => {
    return this.log.child((message) => {
      return {
        ...message,
        context: {
          ...message.context,
          ...getContext(),
        },
      };
    });
  };

  logDebugInfo() {
    // eslint-disable-next-line no-console
    console.log('Canvas manager', this);
    this.log.debug({ manager: this.repr() }, 'Canvas manager');
  }
}
