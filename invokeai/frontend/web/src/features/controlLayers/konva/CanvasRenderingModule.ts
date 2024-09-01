import { CanvasEntityLayerAdapter } from 'features/controlLayers/konva/CanvasEntityLayerAdapter';
import { CanvasEntityMaskAdapter } from 'features/controlLayers/konva/CanvasEntityMaskAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasSessionState } from 'features/controlLayers/store/canvasSessionSlice';
import type { CanvasSettingsState } from 'features/controlLayers/store/canvasSettingsSlice';
import type { CanvasState } from 'features/controlLayers/store/types';
import type { Logger } from 'roarr';

export class CanvasRenderingModule extends CanvasModuleBase {
  readonly type = 'canvas_renderer';

  id: string;
  path: string[];
  log: Logger;
  parent: CanvasManager;
  manager: CanvasManager;

  state: CanvasState | null = null;
  settings: CanvasSettingsState | null = null;
  session: CanvasSessionState | null = null;

  isFirstRender = true;

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('canvas_renderer');
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');
  }

  render = async () => {
    if (!this.state || !this.settings || !this.session) {
      this.log.trace('First render');
    }

    await this.renderCanvas();
    this.renderSettings();
    await this.renderSession();

    // We have no prev state for the first render
    if (this.isFirstRender) {
      this.isFirstRender = false;
      this.manager.setCanvasManager();
    }
  };

  renderCanvas = async () => {
    const state = this.manager.stateApi.getCanvasState();

    const prevState = this.state;
    this.state = state;

    if (prevState === state) {
      // No changes to state - no need to render
      return;
    }

    await this.renderRasterLayers(state, prevState);
    await this.renderControlLayers(prevState, state);
    await this.renderRegionalGuidance(prevState, state);
    await this.renderInpaintMasks(state, prevState);
    await this.renderBbox(state, prevState);
    this.arrangeEntities(state, prevState);

    this.manager.stateApi.$toolState.set(this.manager.stateApi.getToolState());
    this.manager.stateApi.$selectedEntityIdentifier.set(state.selectedEntityIdentifier);
    this.manager.stateApi.$selectedEntity.set(this.manager.stateApi.getSelectedEntity());
    this.manager.stateApi.$currentFill.set(this.manager.stateApi.getCurrentFill());
  };

  renderSettings = () => {
    const settings = this.manager.stateApi.getSettings();

    if (!this.settings) {
      this.log.trace('First settings render');
    }

    const prevSettings = this.settings;
    this.settings = settings;

    if (prevSettings === settings) {
      // No changes to state - no need to render
      return;
    }

    this.renderBackground(settings, prevSettings);
  };

  renderSession = async () => {
    const session = this.manager.stateApi.getSession();

    if (!this.session) {
      this.log.trace('First session render');
    }

    const prevSession = this.session;
    this.session = session;

    if (prevSession === session) {
      // No changes to state - no need to render
      return;
    }

    await this.renderStagingArea(session, prevSession);
  };

  renderBackground = (settings: CanvasSettingsState, prevSettings: CanvasSettingsState | null) => {
    if (!prevSettings || settings.dynamicGrid !== prevSettings.dynamicGrid) {
      this.manager.background.render();
    }
  };

  renderRasterLayers = async (state: CanvasState, prevState: CanvasState | null) => {
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
          adapter = new CanvasEntityLayerAdapter(entityState, this.manager);
          adapterMap.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({ state: entityState });
      }
    }
  };

  renderControlLayers = async (prevState: CanvasState | null, state: CanvasState) => {
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
          adapter = new CanvasEntityLayerAdapter(entityState, this.manager);
          adapterMap.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({ state: entityState });
      }
    }
  };

  renderRegionalGuidance = async (prevState: CanvasState | null, state: CanvasState) => {
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
          adapter = new CanvasEntityMaskAdapter(entityState, this.manager);
          adapterMap.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({ state: entityState });
      }
    }
  };

  renderInpaintMasks = async (state: CanvasState, prevState: CanvasState | null) => {
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
          adapter = new CanvasEntityMaskAdapter(entityState, this.manager);
          adapterMap.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({ state: entityState });
      }
    }
  };

  renderBbox = (state: CanvasState, prevState: CanvasState | null) => {
    if (!prevState || state.bbox !== prevState.bbox) {
      this.manager.bbox.render();
    }
  };

  renderStagingArea = async (session: CanvasSessionState, prevSession: CanvasSessionState | null) => {
    if (!prevSession || session !== prevSession) {
      await this.manager.stagingArea.render();
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
}
