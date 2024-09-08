import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { type CanvasState, getEntityIdentifier } from 'features/controlLayers/store/types';
import type { Logger } from 'roarr';

export class CanvasEntityRendererModule extends CanvasModuleBase {
  readonly type = 'entity_renderer';
  readonly id: string;
  readonly path: string[];
  readonly log: Logger;
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;

  subscriptions = new Set<() => void>();

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('canvas_renderer');
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasSlice, this.render));
  }

  initialize = () => {
    this.log.debug('Initializing module');
    this.render(this.manager.stateApi.runSelector(selectCanvasSlice), null);
  };

  render = async (state: CanvasState, prevState: CanvasState | null) => {
    await this.createNewRasterLayers(state, prevState);
    await this.createNewControlLayers(state, prevState);
    await this.createNewRegionalGuidance(state, prevState);
    await this.createNewInpaintMasks(state, prevState);
    this.arrangeEntities(state, prevState);
  };

  createNewRasterLayers = async (state: CanvasState, prevState: CanvasState | null) => {
    if (!prevState || state.rasterLayers.entities !== prevState.rasterLayers.entities) {
      for (const entityState of state.rasterLayers.entities) {
        if (!this.manager.adapters.rasterLayers.has(entityState.id)) {
          const adapter = this.manager.createAdapter(getEntityIdentifier(entityState));
          await adapter.initialize();
        }
      }
    }
  };

  createNewControlLayers = async (state: CanvasState, prevState: CanvasState | null) => {
    if (!prevState || state.controlLayers.entities !== prevState.controlLayers.entities) {
      for (const entityState of state.controlLayers.entities) {
        if (!this.manager.adapters.controlLayers.has(entityState.id)) {
          const adapter = this.manager.createAdapter(getEntityIdentifier(entityState));
          await adapter.initialize();
        }
      }
    }
  };

  createNewRegionalGuidance = async (state: CanvasState, prevState: CanvasState | null) => {
    if (!prevState || state.regions.entities !== prevState.regions.entities) {
      for (const entityState of state.regions.entities) {
        if (!this.manager.adapters.regionMasks.has(entityState.id)) {
          const adapter = this.manager.createAdapter(getEntityIdentifier(entityState));
          await adapter.initialize();
        }
      }
    }
  };

  createNewInpaintMasks = async (state: CanvasState, prevState: CanvasState | null) => {
    if (!prevState || state.inpaintMasks.entities !== prevState.inpaintMasks.entities) {
      for (const entityState of state.inpaintMasks.entities) {
        if (!this.manager.adapters.inpaintMasks.has(entityState.id)) {
          const adapter = this.manager.createAdapter(getEntityIdentifier(entityState));
          await adapter.initialize();
        }
      }
    }
  };

  arrangeEntities = (state: CanvasState, prevState: CanvasState | null) => {
    if (
      !prevState ||
      state.rasterLayers.entities !== prevState.rasterLayers.entities ||
      state.controlLayers.entities !== prevState.controlLayers.entities ||
      state.regions.entities !== prevState.regions.entities ||
      state.inpaintMasks.entities !== prevState.inpaintMasks.entities ||
      state.selectedEntityIdentifier?.id !== prevState.selectedEntityIdentifier?.id
    ) {
      this.log.trace('Arranging entities');

      let zIndex = 0;

      // Draw order:
      // 1. Background
      // 2. Raster layers
      // 3. Control layers
      // 4. Regions
      // 5. Inpaint masks
      // 6. Preview layer (bbox, staging area, progress image, tool)

      this.manager.background.konva.layer.zIndex(++zIndex);

      for (const { id } of this.manager.stateApi.getRasterLayersState().entities) {
        this.manager.adapters.rasterLayers.get(id)?.konva.layer.zIndex(++zIndex);
      }

      for (const { id } of this.manager.stateApi.getControlLayersState().entities) {
        this.manager.adapters.controlLayers.get(id)?.konva.layer.zIndex(++zIndex);
      }

      for (const { id } of this.manager.stateApi.getRegionsState().entities) {
        this.manager.adapters.regionMasks.get(id)?.konva.layer.zIndex(++zIndex);
      }

      for (const { id } of this.manager.stateApi.getInpaintMasksState().entities) {
        this.manager.adapters.inpaintMasks.get(id)?.konva.layer.zIndex(++zIndex);
      }

      this.manager.konva.previewLayer.zIndex(++zIndex);
    }
  };

  destroy = () => {
    this.log.trace('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
  };
}
