import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import type { SerializableObject } from 'common/types';
import { SyncableMap } from 'common/util/SyncableMap/SyncableMap';
import { CanvasBboxModule } from 'features/controlLayers/konva/CanvasBboxModule';
import { CanvasCacheModule } from 'features/controlLayers/konva/CanvasCacheModule';
import { CanvasCompositorModule } from 'features/controlLayers/konva/CanvasCompositorModule';
import { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterInpaintMask';
import { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRegionalGuidance';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import { CanvasEntityRendererModule } from 'features/controlLayers/konva/CanvasEntityRendererModule';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasProgressImageModule } from 'features/controlLayers/konva/CanvasProgressImageModule';
import { CanvasStageModule } from 'features/controlLayers/konva/CanvasStageModule';
import { CanvasStagingAreaModule } from 'features/controlLayers/konva/CanvasStagingAreaModule';
import { CanvasToolModule } from 'features/controlLayers/konva/CanvasTool/CanvasToolModule';
import { CanvasWorkerModule } from 'features/controlLayers/konva/CanvasWorkerModule.js';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { $canvasManager } from 'features/controlLayers/store/ephemeral';
import type { CanvasEntityIdentifier, CanvasEntityType } from 'features/controlLayers/store/types';
import {
  isControlLayerEntityIdentifier,
  isInpaintMaskEntityIdentifier,
  isRasterLayerEntityIdentifier,
  isRegionalGuidanceEntityIdentifier,
} from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Atom } from 'nanostores';
import { computed } from 'nanostores';
import type { Logger } from 'roarr';
import type { AppSocket } from 'services/events/types';
import { assert } from 'tsafe';

import { CanvasBackgroundModule } from './CanvasBackgroundModule';
import { CanvasStateApiModule } from './CanvasStateApiModule';

export class CanvasManager extends CanvasModuleBase {
  readonly type = 'manager';
  readonly id: string;
  readonly path: string[];
  readonly manager: CanvasManager;
  readonly parent: CanvasManager;
  readonly log: Logger;

  store: AppStore;
  socket: AppSocket;

  adapters = {
    rasterLayers: new SyncableMap<string, CanvasEntityAdapterRasterLayer>(),
    controlLayers: new SyncableMap<string, CanvasEntityAdapterControlLayer>(),
    regionMasks: new SyncableMap<string, CanvasEntityAdapterRegionalGuidance>(),
    inpaintMasks: new SyncableMap<string, CanvasEntityAdapterInpaintMask>(),
  };

  stateApi: CanvasStateApiModule;
  background: CanvasBackgroundModule;
  stage: CanvasStageModule;
  worker: CanvasWorkerModule;
  cache: CanvasCacheModule;
  entityRenderer: CanvasEntityRendererModule;
  compositor: CanvasCompositorModule;
  tool: CanvasToolModule;
  bbox: CanvasBboxModule;
  stagingArea: CanvasStagingAreaModule;
  progressImage: CanvasProgressImageModule;

  konva: {
    previewLayer: Konva.Layer;
  };

  _isDebugging: boolean = false;

  /**
   * Whether the canvas is currently busy with a transformation, filter, rasterization, staging or compositing operation.
   */
  $isBusy: Atom<boolean>;

  constructor(container: HTMLDivElement, store: AppStore, socket: AppSocket) {
    super();
    this.id = getPrefixedId(this.type);
    this.path = [this.id];
    this.manager = this;
    this.parent = this;
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
    this.stage = new CanvasStageModule(container, this);
    this.worker = new CanvasWorkerModule(this);
    this.cache = new CanvasCacheModule(this);
    this.entityRenderer = new CanvasEntityRendererModule(this);

    this.compositor = new CanvasCompositorModule(this);
    this.stagingArea = new CanvasStagingAreaModule(this);

    this.$isBusy = computed(
      [
        this.stateApi.$isFiltering,
        this.stateApi.$isTransforming,
        this.stateApi.$isRasterizing,
        this.stagingArea.$isStaging,
        this.compositor.$isBusy,
      ],
      (isFiltering, isTransforming, isRasterizing, isStaging, isCompositing) => {
        return isFiltering || isTransforming || isRasterizing || isStaging || isCompositing;
      }
    );

    this.background = new CanvasBackgroundModule(this);
    this.stage.addLayer(this.background.konva.layer);

    this.konva = {
      previewLayer: new Konva.Layer({ listening: false, imageSmoothingEnabled: false }),
    };
    this.stage.addLayer(this.konva.previewLayer);

    this.tool = new CanvasToolModule(this);
    this.progressImage = new CanvasProgressImageModule(this);
    this.bbox = new CanvasBboxModule(this);

    // Must add in this order for correct z-index
    this.konva.previewLayer.add(this.stagingArea.konva.group);
    this.konva.previewLayer.add(this.progressImage.konva.group);
    this.konva.previewLayer.add(this.bbox.konva.group);
    this.konva.previewLayer.add(this.tool.konva.group);
  }

  getAdapter = <T extends CanvasEntityType = CanvasEntityType>(
    entityIdentifier: CanvasEntityIdentifier<T>
  ): Extract<CanvasEntityAdapter, { state: { type: T } }> | null => {
    switch (entityIdentifier.type) {
      case 'raster_layer':
        return (
          (this.adapters.rasterLayers.get(entityIdentifier.id) as Extract<
            CanvasEntityAdapter,
            { state: { type: T } }
          >) ?? null
        );
      case 'control_layer':
        return (
          (this.adapters.controlLayers.get(entityIdentifier.id) as Extract<
            CanvasEntityAdapter,
            { state: { type: T } }
          >) ?? null
        );
      case 'regional_guidance':
        return (
          (this.adapters.regionMasks.get(entityIdentifier.id) as Extract<
            CanvasEntityAdapter,
            { state: { type: T } }
          >) ?? null
        );
      case 'inpaint_mask':
        return (
          (this.adapters.inpaintMasks.get(entityIdentifier.id) as Extract<
            CanvasEntityAdapter,
            { state: { type: T } }
          >) ?? null
        );
      default:
        return null;
    }
  };

  deleteAdapter = (entityIdentifier: CanvasEntityIdentifier): boolean => {
    switch (entityIdentifier.type) {
      case 'raster_layer':
        return this.adapters.rasterLayers.delete(entityIdentifier.id);
      case 'control_layer':
        return this.adapters.controlLayers.delete(entityIdentifier.id);
      case 'regional_guidance':
        return this.adapters.regionMasks.delete(entityIdentifier.id);
      case 'inpaint_mask':
        return this.adapters.inpaintMasks.delete(entityIdentifier.id);
      default:
        return false;
    }
  };

  getAllAdapters = (): CanvasEntityAdapter[] => {
    return [
      ...this.adapters.rasterLayers.values(),
      ...this.adapters.controlLayers.values(),
      ...this.adapters.regionMasks.values(),
      ...this.adapters.inpaintMasks.values(),
    ];
  };

  createAdapter = (entityIdentifier: CanvasEntityIdentifier): CanvasEntityAdapter => {
    if (isRasterLayerEntityIdentifier(entityIdentifier)) {
      const adapter = new CanvasEntityAdapterRasterLayer(entityIdentifier, this);
      this.adapters.rasterLayers.set(adapter.id, adapter);
      return adapter;
    } else if (isControlLayerEntityIdentifier(entityIdentifier)) {
      const adapter = new CanvasEntityAdapterControlLayer(entityIdentifier, this);
      this.adapters.controlLayers.set(adapter.id, adapter);
      return adapter;
    } else if (isRegionalGuidanceEntityIdentifier(entityIdentifier)) {
      const adapter = new CanvasEntityAdapterRegionalGuidance(entityIdentifier, this);
      this.adapters.regionMasks.set(adapter.id, adapter);
      return adapter;
    } else if (isInpaintMaskEntityIdentifier(entityIdentifier)) {
      const adapter = new CanvasEntityAdapterInpaintMask(entityIdentifier, this);
      this.adapters.inpaintMasks.set(adapter.id, adapter);
      return adapter;
    } else {
      assert(false, 'Unhandled entity type');
    }
  };

  enableDebugging() {
    this._isDebugging = true;
    this.logDebugInfo();
  }

  disableDebugging() {
    this._isDebugging = false;
  }

  getAllModules = (): CanvasModuleBase[] => {
    return [
      this.bbox,
      this.stagingArea,
      this.tool,
      this.progressImage,
      this.stateApi,
      this.background,
      this.worker,
      this.entityRenderer,
      this.compositor,
      this.stage,
    ];
  };

  initialize = () => {
    this.log.debug('Initializing');

    for (const canvasModule of this.getAllModules()) {
      canvasModule.initialize?.();
    }

    $canvasManager.set(this);
  };

  destroy = () => {
    this.log.debug('Destroying module');

    for (const adapter of this.getAllAdapters()) {
      adapter.destroy();
    }

    for (const canvasModule of this.getAllModules()) {
      canvasModule.destroy();
    }

    $canvasManager.set(null);
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      $isBusy: this.$isBusy.get(),
      rasterLayers: Array.from(this.adapters.rasterLayers.values()).map((adapter) => adapter.repr()),
      controlLayers: Array.from(this.adapters.controlLayers.values()).map((adapter) => adapter.repr()),
      inpaintMasks: Array.from(this.adapters.inpaintMasks.values()).map((adapter) => adapter.repr()),
      regionMasks: Array.from(this.adapters.regionMasks.values()).map((adapter) => adapter.repr()),
      stateApi: this.stateApi.repr(),
      bbox: this.bbox.repr(),
      stagingArea: this.stagingArea.repr(),
      tool: this.tool.repr(),
      progressImage: this.progressImage.repr(),
      background: this.background.repr(),
      worker: this.worker.repr(),
      entityRenderer: this.entityRenderer.repr(),
      compositor: this.compositor.repr(),
      stage: this.stage.repr(),
    };
  };

  getLoggingContext = (): SerializableObject => ({ path: this.path });

  buildPath = (canvasModule: CanvasModuleBase): string[] => {
    return canvasModule.parent.path.concat(canvasModule.id);
  };

  buildLogger = (canvasModule: CanvasModuleBase): Logger => {
    return this.log.child((message) => {
      return {
        ...message,
        context: {
          ...message.context,
          ...canvasModule.getLoggingContext(),
        },
      };
    });
  };

  logDebugInfo() {
    /**
     * We are logging the live manager instance here, so we cannot use the logger, which only accepts serializable
     * objects.
     */
    // eslint-disable-next-line no-console
    console.log('Canvas manager', { managerInstance: this, managerInfo: this.repr() });
  }
}
