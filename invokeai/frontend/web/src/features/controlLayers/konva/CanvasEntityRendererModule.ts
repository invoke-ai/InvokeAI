import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectCanvasSlice,
  selectControlLayerEntities,
  selectInpaintMaskEntities,
  selectRasterLayerEntities,
  selectRegionalGuidanceEntities,
} from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  CanvasState,
} from 'features/controlLayers/store/types';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
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

    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectRasterLayerEntities, this.createNewRasterLayers)
    );

    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectControlLayerEntities, this.createNewControlLayers)
    );

    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectInpaintMaskEntities, this.createNewInpaintMasks)
    );

    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectRegionalGuidanceEntities, this.createNewRegionalGuidance)
    );

    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectCanvasSlice, this.arrangeEntities));
  }

  initialize = () => {
    this.log.debug('Initializing module');
    this.createNewRasterLayers(this.manager.stateApi.runSelector(selectRasterLayerEntities));
    this.createNewControlLayers(this.manager.stateApi.runSelector(selectControlLayerEntities));
    this.createNewRegionalGuidance(this.manager.stateApi.runSelector(selectRegionalGuidanceEntities));
    this.createNewInpaintMasks(this.manager.stateApi.runSelector(selectInpaintMaskEntities));
    this.arrangeEntities(this.manager.stateApi.runSelector(selectCanvasSlice), null);
  };

  createNewRasterLayers = (entities: CanvasRasterLayerState[]) => {
    for (const entityState of entities) {
      if (!this.manager.adapters.rasterLayers.has(entityState.id)) {
        const adapter = this.manager.createAdapter(getEntityIdentifier(entityState));
        adapter.initialize();
      }
    }
  };

  createNewControlLayers = (entities: CanvasControlLayerState[]) => {
    for (const entityState of entities) {
      if (!this.manager.adapters.controlLayers.has(entityState.id)) {
        const adapter = this.manager.createAdapter(getEntityIdentifier(entityState));
        adapter.initialize();
      }
    }
  };

  createNewRegionalGuidance = (entities: CanvasRegionalGuidanceState[]) => {
    for (const entityState of entities) {
      if (!this.manager.adapters.regionMasks.has(entityState.id)) {
        const adapter = this.manager.createAdapter(getEntityIdentifier(entityState));
        adapter.initialize();
      }
    }
  };

  createNewInpaintMasks = (entities: CanvasInpaintMaskState[]) => {
    for (const entityState of entities) {
      if (!this.manager.adapters.inpaintMasks.has(entityState.id)) {
        const adapter = this.manager.createAdapter(getEntityIdentifier(entityState));
        adapter.initialize();
      }
    }
  };

  arrangeEntities = (state: CanvasState, prevState: CanvasState | null) => {
    if (
      !prevState ||
      state.rasterLayers.entities !== prevState.rasterLayers.entities ||
      state.controlLayers.entities !== prevState.controlLayers.entities ||
      state.regionalGuidance.entities !== prevState.regionalGuidance.entities ||
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

      this.manager.background.konva.layer.zIndex(zIndex++);

      for (const { id } of this.manager.stateApi.getRasterLayersState().entities) {
        this.manager.adapters.rasterLayers.get(id)?.konva.layer.zIndex(zIndex++);
      }

      for (const { id } of this.manager.stateApi.getControlLayersState().entities) {
        this.manager.adapters.controlLayers.get(id)?.konva.layer.zIndex(zIndex++);
      }

      for (const { id } of this.manager.stateApi.getRegionsState().entities) {
        this.manager.adapters.regionMasks.get(id)?.konva.layer.zIndex(zIndex++);
      }

      for (const { id } of this.manager.stateApi.getInpaintMasksState().entities) {
        this.manager.adapters.inpaintMasks.get(id)?.konva.layer.zIndex(zIndex++);
      }

      this.manager.konva.previewLayer.zIndex(zIndex++);
    }
  };

  destroy = () => {
    this.log.trace('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
  };
}
