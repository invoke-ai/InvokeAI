import type { SerializableObject } from 'common/types';
import { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import type { Logger } from 'roarr';

export class CanvasRenderingModule {
  id: string;
  path: string[];
  log: Logger;
  manager: CanvasManager;

  state: CanvasV2State | null = null;

  constructor(manager: CanvasManager) {
    this.id = getPrefixedId('canvas_renderer');
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug('Creating canvas renderer');
  }

  render = async () => {
    const state = this.manager.stateApi.getState();

    if (!this.state) {
      this.log.trace('First render');
    }

    const prevState = this.state;
    this.state = state;

    if (prevState === state) {
      // No changes to state - no need to render
      return;
    }

    this.renderBackground(state, prevState);
    await this.renderRasterLayers(state, prevState);
    await this.renderControlLayers(prevState, state);
    await this.renderRegionalGuidance(prevState, state);
    await this.renderInpaintMasks(state, prevState);
    await this.renderBbox(state, prevState);
    await this.renderStagingArea(state, prevState);
    this.arrangeEntities(state, prevState);

    this.manager.stateApi.$toolState.set(state.tool);
    this.manager.stateApi.$selectedEntityIdentifier.set(state.selectedEntityIdentifier);
    this.manager.stateApi.$selectedEntity.set(this.manager.stateApi.getSelectedEntity());
    this.manager.stateApi.$currentFill.set(this.manager.stateApi.getCurrentFill());

    // We have no prev state for the first render
    if (!prevState) {
      this.manager.setCanvasManager();
    }
  };

  getLoggingContext = (): SerializableObject => {
    return { ...this.manager.getLoggingContext(), path: this.manager.path.join('.') };
  };

  renderBackground = (state: CanvasV2State, prevState: CanvasV2State | null) => {
    if (!prevState || state.settings.dynamicGrid !== prevState.settings.dynamicGrid) {
      this.manager.background.render();
    }
  };

  renderRasterLayers = async (state: CanvasV2State, prevState: CanvasV2State | null) => {
    if (!prevState || state.rasterLayers.isHidden !== prevState.rasterLayers.isHidden) {
      for (const adapter of this.manager.rasterLayerAdapters.values()) {
        adapter.renderer.updateOpacity(state.rasterLayers.isHidden ? 0 : adapter.state.opacity);
      }
    }

    if (!prevState || state.rasterLayers.entities !== prevState.rasterLayers.entities) {
      for (const entityAdapter of this.manager.rasterLayerAdapters.values()) {
        if (!state.rasterLayers.entities.find((l) => l.id === entityAdapter.id)) {
          await entityAdapter.destroy();
          this.manager.rasterLayerAdapters.delete(entityAdapter.id);
        }
      }

      for (const entityState of state.rasterLayers.entities) {
        let adapter = this.manager.rasterLayerAdapters.get(entityState.id);
        if (!adapter) {
          adapter = new CanvasLayerAdapter(entityState, this.manager);
          this.manager.rasterLayerAdapters.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({
          state: entityState,
          toolState: state.tool,
          isSelected: state.selectedEntityIdentifier?.id === entityState.id,
        });
      }
    }
  };

  renderControlLayers = async (prevState: CanvasV2State | null, state: CanvasV2State) => {
    if (!prevState || state.controlLayers.isHidden !== prevState.controlLayers.isHidden) {
      for (const adapter of this.manager.controlLayerAdapters.values()) {
        adapter.renderer.updateOpacity(state.controlLayers.isHidden ? 0 : adapter.state.opacity);
      }
    }

    if (!prevState || state.controlLayers.entities !== prevState.controlLayers.entities) {
      for (const entityAdapter of this.manager.controlLayerAdapters.values()) {
        if (!state.controlLayers.entities.find((l) => l.id === entityAdapter.id)) {
          await entityAdapter.destroy();
          this.manager.controlLayerAdapters.delete(entityAdapter.id);
        }
      }

      for (const entityState of state.controlLayers.entities) {
        let adapter = this.manager.controlLayerAdapters.get(entityState.id);
        if (!adapter) {
          adapter = new CanvasLayerAdapter(entityState, this.manager);
          this.manager.controlLayerAdapters.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({
          state: entityState,
          toolState: state.tool,
          isSelected: state.selectedEntityIdentifier?.id === entityState.id,
        });
      }
    }
  };

  renderRegionalGuidance = async (prevState: CanvasV2State | null, state: CanvasV2State) => {
    if (!prevState || state.regions.isHidden !== prevState.regions.isHidden) {
      for (const adapter of this.manager.regionalGuidanceAdapters.values()) {
        adapter.renderer.updateOpacity(state.regions.isHidden ? 0 : adapter.state.opacity);
      }
    }

    if (
      !prevState ||
      state.regions.entities !== prevState.regions.entities ||
      state.tool.selected !== prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== prevState.selectedEntityIdentifier?.id
    ) {
      // Destroy the konva nodes for nonexistent entities
      for (const canvasRegion of this.manager.regionalGuidanceAdapters.values()) {
        if (!state.regions.entities.find((rg) => rg.id === canvasRegion.id)) {
          canvasRegion.destroy();
          this.manager.regionalGuidanceAdapters.delete(canvasRegion.id);
        }
      }

      for (const entityState of state.regions.entities) {
        let adapter = this.manager.regionalGuidanceAdapters.get(entityState.id);
        if (!adapter) {
          adapter = new CanvasMaskAdapter(entityState, this.manager);
          this.manager.regionalGuidanceAdapters.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({
          state: entityState,
          toolState: state.tool,
          isSelected: state.selectedEntityIdentifier?.id === entityState.id,
        });
      }
    }
  };

  renderInpaintMasks = async (state: CanvasV2State, prevState: CanvasV2State | null) => {
    if (!prevState || state.inpaintMasks.isHidden !== prevState.inpaintMasks.isHidden) {
      for (const adapter of this.manager.inpaintMaskAdapters.values()) {
        adapter.renderer.updateOpacity(state.inpaintMasks.isHidden ? 0 : adapter.state.opacity);
      }
    }

    if (
      !prevState ||
      state.inpaintMasks.entities !== prevState.inpaintMasks.entities ||
      state.tool.selected !== prevState.tool.selected ||
      state.selectedEntityIdentifier?.id !== prevState.selectedEntityIdentifier?.id
    ) {
      // Destroy the konva nodes for nonexistent entities
      for (const adapter of this.manager.inpaintMaskAdapters.values()) {
        if (!state.inpaintMasks.entities.find((rg) => rg.id === adapter.id)) {
          adapter.destroy();
          this.manager.inpaintMaskAdapters.delete(adapter.id);
        }
      }

      for (const entityState of state.inpaintMasks.entities) {
        let adapter = this.manager.inpaintMaskAdapters.get(entityState.id);
        if (!adapter) {
          adapter = new CanvasMaskAdapter(entityState, this.manager);
          this.manager.inpaintMaskAdapters.set(adapter.id, adapter);
          this.manager.stage.addLayer(adapter.konva.layer);
        }
        await adapter.update({
          state: entityState,
          toolState: state.tool,
          isSelected: state.selectedEntityIdentifier?.id === entityState.id,
        });
      }
    }
  };

  renderBbox = (state: CanvasV2State, prevState: CanvasV2State | null) => {
    if (!prevState || state.bbox !== prevState.bbox || state.tool.selected !== prevState.tool.selected) {
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
        this.manager.rasterLayerAdapters.get(id)?.konva.layer.zIndex(++zIndex);
      }

      for (const { id } of this.manager.stateApi.getControlLayersState().entities) {
        this.manager.controlLayerAdapters.get(id)?.konva.layer.zIndex(++zIndex);
      }

      for (const { id } of this.manager.stateApi.getRegionsState().entities) {
        this.manager.regionalGuidanceAdapters.get(id)?.konva.layer.zIndex(++zIndex);
      }

      for (const { id } of this.manager.stateApi.getInpaintMasksState().entities) {
        this.manager.inpaintMaskAdapters.get(id)?.konva.layer.zIndex(++zIndex);
      }

      this.manager.preview.getLayer().zIndex(++zIndex);
    }
  };
}
