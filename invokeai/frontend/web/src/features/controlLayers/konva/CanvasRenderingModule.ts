import type { SerializableObject } from 'common/types';
import { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasSettingsState } from 'features/controlLayers/store/canvasSettingsSlice';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import type { Logger } from 'roarr';

export class CanvasRenderingModule extends CanvasModuleBase {
  readonly type = 'canvas_renderer';

  id: string;
  path: string[];
  log: Logger;
  manager: CanvasManager;
  subscriptions = new Set<() => void>();

  state: CanvasV2State | null = null;
  settings: CanvasSettingsState | null = null;

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('canvas_renderer');
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug('Creating canvas renderer module');
  }

  render = async () => {
    const state = this.manager.stateApi.getCanvasState();
    const settings = this.manager.stateApi.getSettings();

    if (!this.state || !this.settings) {
      this.log.trace('First render');
    }

    const prevState = this.state;
    this.state = state;

    const prevSettings = this.settings;
    this.settings = settings;

    if (prevState === state && prevSettings === settings) {
      // No changes to state - no need to render
      return;
    }

    this.renderBackground(settings, prevSettings);
    await this.renderRasterLayers(state, prevState);
    await this.renderControlLayers(prevState, state);
    await this.renderRegionalGuidance(prevState, state);
    await this.renderInpaintMasks(state, prevState);
    await this.renderBbox(state, prevState);
    await this.renderStagingArea(state, prevState);
    this.arrangeEntities(state, prevState);

    this.manager.stateApi.$toolState.set(this.manager.stateApi.getToolState());
    this.manager.stateApi.$selectedEntityIdentifier.set(state.selectedEntityIdentifier);
    this.manager.stateApi.$selectedEntity.set(this.manager.stateApi.getSelectedEntity());
    this.manager.stateApi.$currentFill.set(this.manager.stateApi.getCurrentFill());

    // We have no prev state for the first render
    if (!prevState && !prevSettings) {
      this.manager.setCanvasManager();
    }
  };

  getLoggingContext = (): SerializableObject => {
    return { ...this.manager.getLoggingContext(), path: this.manager.path.join('.') };
  };

  renderBackground = (settings: CanvasSettingsState, prevSettings: CanvasSettingsState | null) => {
    if (!prevSettings || settings.dynamicGrid !== prevSettings.dynamicGrid) {
      this.manager.background.render();
    }
  };

  renderRasterLayers = async (state: CanvasV2State, prevState: CanvasV2State | null) => {
    const adapterMap = this.manager.adapters.rasterLayers;

    if (!prevState || state.rasterLayers.isHidden !== prevState.rasterLayers.isHidden) {
      for (const adapter of adapterMap.values()) {
        adapter.renderer.updateOpacity(state.rasterLayers.isHidden ? 0 : adapter.state.opacity);
      }
    }

    if (!prevState || state.rasterLayers.entities !== prevState.rasterLayers.entities) {
      for (const entityAdapter of adapterMap.values()) {
        if (!state.rasterLayers.entities.find((l) => l.id === entityAdapter.id)) {
          await entityAdapter.destroy();
          adapterMap.delete(entityAdapter.id);
        }
      }

      for (const entityState of state.rasterLayers.entities) {
        let adapter = adapterMap.get(entityState.id);
        if (!adapter) {
          adapter = new CanvasLayerAdapter(entityState, this.manager);
          adapterMap.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({ state: entityState });
      }
    }
  };

  renderControlLayers = async (prevState: CanvasV2State | null, state: CanvasV2State) => {
    const adapterMap = this.manager.adapters.controlLayers;

    if (!prevState || state.controlLayers.isHidden !== prevState.controlLayers.isHidden) {
      for (const adapter of adapterMap.values()) {
        adapter.renderer.updateOpacity(state.controlLayers.isHidden ? 0 : adapter.state.opacity);
      }
    }

    if (!prevState || state.controlLayers.entities !== prevState.controlLayers.entities) {
      for (const entityAdapter of adapterMap.values()) {
        if (!state.controlLayers.entities.find((l) => l.id === entityAdapter.id)) {
          await entityAdapter.destroy();
          adapterMap.delete(entityAdapter.id);
        }
      }

      for (const entityState of state.controlLayers.entities) {
        let adapter = adapterMap.get(entityState.id);
        if (!adapter) {
          adapter = new CanvasLayerAdapter(entityState, this.manager);
          adapterMap.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({ state: entityState });
      }
    }
  };

  renderRegionalGuidance = async (prevState: CanvasV2State | null, state: CanvasV2State) => {
    const adapterMap = this.manager.adapters.regionMasks;

    if (!prevState || state.regions.isHidden !== prevState.regions.isHidden) {
      for (const adapter of adapterMap.values()) {
        adapter.renderer.updateOpacity(state.regions.isHidden ? 0 : adapter.state.opacity);
      }
    }

    if (
      !prevState ||
      state.regions.entities !== prevState.regions.entities ||
      state.selectedEntityIdentifier?.id !== prevState.selectedEntityIdentifier?.id
    ) {
      // Destroy the konva nodes for nonexistent entities
      for (const canvasRegion of adapterMap.values()) {
        if (!state.regions.entities.find((rg) => rg.id === canvasRegion.id)) {
          canvasRegion.destroy();
          adapterMap.delete(canvasRegion.id);
        }
      }

      for (const entityState of state.regions.entities) {
        let adapter = adapterMap.get(entityState.id);
        if (!adapter) {
          adapter = new CanvasMaskAdapter(entityState, this.manager);
          adapterMap.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({ state: entityState });
      }
    }
  };

  renderInpaintMasks = async (state: CanvasV2State, prevState: CanvasV2State | null) => {
    const adapterMap = this.manager.adapters.inpaintMasks;

    if (!prevState || state.inpaintMasks.isHidden !== prevState.inpaintMasks.isHidden) {
      for (const adapter of adapterMap.values()) {
        adapter.renderer.updateOpacity(state.inpaintMasks.isHidden ? 0 : adapter.state.opacity);
      }
    }

    if (
      !prevState ||
      state.inpaintMasks.entities !== prevState.inpaintMasks.entities ||
      state.selectedEntityIdentifier?.id !== prevState.selectedEntityIdentifier?.id
    ) {
      // Destroy the konva nodes for nonexistent entities
      for (const adapter of adapterMap.values()) {
        if (!state.inpaintMasks.entities.find((rg) => rg.id === adapter.id)) {
          adapter.destroy();
          adapterMap.delete(adapter.id);
        }
      }

      for (const entityState of state.inpaintMasks.entities) {
        let adapter = adapterMap.get(entityState.id);
        if (!adapter) {
          adapter = new CanvasMaskAdapter(entityState, this.manager);
          adapterMap.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({ state: entityState });
      }
    }
  };

  renderBbox = (state: CanvasV2State, prevState: CanvasV2State | null) => {
    if (!prevState || state.bbox !== prevState.bbox) {
      this.manager.preview.bbox.render();
    }
  };

  renderStagingArea = async (state: CanvasV2State, prevState: CanvasV2State | null) => {
    if (!prevState || state.session !== prevState.session) {
      await this.manager.preview.stagingArea.render();
    }
  };

  arrangeEntities = (state: CanvasV2State, prevState: CanvasV2State | null) => {
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
      // 6. Preview (bbox, staging area, progress image, tool)

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

      this.manager.preview.getLayer().zIndex(++zIndex);
    }
  };

  repr = () => {
    return {
      id: this.id,
      path: this.path,
      type: this.type,
    };
  };

  destroy = () => {
    this.log.debug('Destroying canvas renderer module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
  };
}
