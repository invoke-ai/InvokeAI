import type { AppSocket } from 'app/hooks/useSocketIO';
import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import type { SerializableObject } from 'common/types';
import { CanvasCacheModule } from 'features/controlLayers/konva/CanvasCacheModule';
import { CanvasCompositorModule } from 'features/controlLayers/konva/CanvasCompositorModule';
import { CanvasFilterModule } from 'features/controlLayers/konva/CanvasFilterModule';
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
import { setStageEventHandlers } from './events';

export const $canvasManager = atom<CanvasManager | null>(null);
const TYPE = 'manager';

export class CanvasManager {
  readonly type = TYPE;

  id: string;
  path: string[];

  store: AppStore;
  socket: AppSocket;

  rasterLayerAdapters: Map<string, CanvasLayerAdapter> = new Map();
  controlLayerAdapters: Map<string, CanvasLayerAdapter> = new Map();
  regionalGuidanceAdapters: Map<string, CanvasMaskAdapter> = new Map();
  inpaintMaskAdapters: Map<string, CanvasMaskAdapter> = new Map();

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
    this.id = getPrefixedId(this.type);
    this.path = [this.id];
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

  log = logger('canvas').child((message) => {
    return {
      ...message,
      context: {
        ...this.getLoggingContext(),
        ...message.context,
      },
    };
  });

  enableDebugging() {
    this._isDebugging = true;
    this.logDebugInfo();
  }

  disableDebugging() {
    this._isDebugging = false;
  }

  initialize = () => {
    this.log.debug('Initializing canvas manager');

    // These atoms require the canvas manager to be set up before we can provide their initial values
    this.stateApi.$transformingEntity.set(null);
    this.stateApi.$toolState.set(this.stateApi.getToolState());
    this.stateApi.$selectedEntityIdentifier.set(this.stateApi.getState().selectedEntityIdentifier);
    this.stateApi.$currentFill.set(this.stateApi.getCurrentFill());
    this.stateApi.$selectedEntity.set(this.stateApi.getSelectedEntity());

    const cleanupEventHandlers = setStageEventHandlers(this);
    const cleanupStage = this.stage.initialize();
    const cleanupStore = this.store.subscribe(this.renderer.render);

    return () => {
      this.log.debug('Cleaning up canvas manager');
      const allAdapters = [
        ...this.rasterLayerAdapters.values(),
        ...this.controlLayerAdapters.values(),
        ...this.inpaintMaskAdapters.values(),
        ...this.regionalGuidanceAdapters.values(),
      ];
      for (const adapter of allAdapters) {
        adapter.destroy();
      }
      this.background.destroy();
      this.preview.destroy();
      cleanupStore();
      cleanupEventHandlers();
      cleanupStage();
    };
  };

  setCanvasManager = () => {
    this.log.debug('Setting canvas manager');
    $canvasManager.set(this);
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
    console.log(this);
    for (const layer of this.rasterLayerAdapters.values()) {
      // eslint-disable-next-line no-console
      console.log(layer);
    }
  }

  getPrefixedId = getPrefixedId;
}
