import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { type CanvasState, getEntityIdentifier } from 'features/controlLayers/store/types';
import type { Logger } from 'roarr';

export class CanvasEntityRendererModule extends CanvasModuleBase {
  readonly type = 'entity_renderer';
  readonly id: string;
  readonly path: string[];
  readonly log: Logger;
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;

  private _state: CanvasState | null = null;

  subscriptions = new Set<() => void>();

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('canvas_renderer');
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.subscriptions.add(this.manager.stateApi.store.subscribe(this.render));
  }

  render = () => {
    const state = this.manager.stateApi.getCanvasState();

    const prevState = this._state;
    this._state = state;

    this.manager.stateApi.$settingsState.set(this.manager.stateApi.getSettings());
    this.manager.stateApi.$selectedEntityIdentifier.set(state.selectedEntityIdentifier);
    this.manager.stateApi.$currentFill.set(this.manager.stateApi.getCurrentColor());

    if (prevState === state) {
      // No changes to state - no need to render
      return;
    }

    this.renderRasterLayers(state, prevState);
    this.renderControlLayers(prevState, state);
    this.renderRegionalGuidance(prevState, state);
    this.renderInpaintMasks(state, prevState);
    this.arrangeEntities(state, prevState);
    this.manager.tool.syncCursorStyle();
  };

  renderRasterLayers = (state: CanvasState, prevState: CanvasState | null) => {
    const adapterMap = this.manager.adapters.rasterLayers;

    if (!prevState || state.rasterLayers.isHidden !== prevState.rasterLayers.isHidden) {
      for (const adapter of adapterMap.values()) {
        adapter.syncOpacity();
      }
    }

    if (!prevState || state.rasterLayers.entities !== prevState.rasterLayers.entities) {
      for (const entityState of state.rasterLayers.entities) {
        if (!adapterMap.has(entityState.id)) {
          this.manager.createAdapter(getEntityIdentifier(entityState));
        }
      }
    }
  };

  renderControlLayers = (prevState: CanvasState | null, state: CanvasState) => {
    const adapterMap = this.manager.adapters.controlLayers;

    if (!prevState || state.controlLayers.isHidden !== prevState.controlLayers.isHidden) {
      for (const adapter of adapterMap.values()) {
        adapter.syncOpacity();
      }
    }

    if (!prevState || state.controlLayers.entities !== prevState.controlLayers.entities) {
      for (const entityState of state.controlLayers.entities) {
        if (!adapterMap.has(entityState.id)) {
          this.manager.createAdapter(getEntityIdentifier(entityState));
        }
      }
    }
  };

  renderRegionalGuidance = (prevState: CanvasState | null, state: CanvasState) => {
    const adapterMap = this.manager.adapters.regionMasks;

    if (!prevState || state.regions.isHidden !== prevState.regions.isHidden) {
      for (const adapter of adapterMap.values()) {
        adapter.syncOpacity();
      }
    }

    if (!prevState || state.regions.entities !== prevState.regions.entities) {
      for (const entityState of state.regions.entities) {
        if (!adapterMap.has(entityState.id)) {
          this.manager.createAdapter(getEntityIdentifier(entityState));
        }
      }
    }
  };

  renderInpaintMasks = (state: CanvasState, prevState: CanvasState | null) => {
    const adapterMap = this.manager.adapters.inpaintMasks;

    if (!prevState || state.inpaintMasks.isHidden !== prevState.inpaintMasks.isHidden) {
      for (const adapter of adapterMap.values()) {
        adapter.syncOpacity();
      }
    }

    if (!prevState || state.inpaintMasks.entities !== prevState.inpaintMasks.entities) {
      for (const entityState of state.inpaintMasks.entities) {
        if (!adapterMap.has(entityState.id)) {
          this.manager.createAdapter(getEntityIdentifier(entityState));
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
      this.log.debug('Arranging entities');

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
