import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { Selector } from '@reduxjs/toolkit';
import type { AppStore, RootState } from 'app/store/store';
import { addAppListener } from 'app/store/store';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { SubscriptionHandler } from 'features/controlLayers/konva/util';
import { createReduxSubscription, getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectCanvasSettingsSlice,
  settingsBgColorChanged,
  settingsBrushWidthChanged,
  settingsEraserWidthChanged,
  settingsFgColorChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import {
  bboxChangedFromCanvas,
  controlLayerAdded,
  entityBrushLineAdded,
  entityEraserLineAdded,
  entityMovedBy,
  entityMovedTo,
  entityRasterized,
  entityRectAdded,
  entityReset,
  inpaintMaskAdded,
  rasterLayerAdded,
  rgAdded,
} from 'features/controlLayers/store/canvasInstanceSlice';
import { selectCanvasSessionSlice } from 'features/controlLayers/store/canvasStagingAreaSlice';
import {
  selectAllRenderableEntities,
  selectBbox,
  selectCanvasSlice,
  selectGridSize,
} from 'features/controlLayers/store/selectors';
import type {
  CanvasState,
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedByPayload,
  EntityMovedToPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
  Rect,
  RgbaColor,
} from 'features/controlLayers/store/types';
import { RGBA_BLACK } from 'features/controlLayers/store/types';
import { zImageOutput } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { getImageDTO } from 'services/api/endpoints/images';
import type { RunGraphOptions } from 'services/api/run-graph';
import { buildRunGraphDependencies, runGraph } from 'services/api/run-graph';
import type { ImageDTO, S } from 'services/api/types';
import type { Param0 } from 'tsafe';

import type { CanvasEntityAdapter } from './CanvasEntity/types';

export class CanvasStateApiModule extends CanvasModuleBase {
  readonly type = 'state_api';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  /**
   * The redux store.
   */
  store: AppStore;

  constructor(store: AppStore, manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating state api module');

    this.store = store;
  }

  /**
   * Runs a selector on the redux store.
   */
  runSelector = <T>(selector: Selector<RootState, T>) => {
    return selector(this.store.getState());
  };

  /**
   * Creates a subscription to the redux store.
   */
  createStoreSubscription = <T>(selector: Selector<RootState, T>, handler: SubscriptionHandler<T>) => {
    return createReduxSubscription(this.store, selector, handler);
  };

  /**
   * Adds a redux listener middleware listener.
   *
   * TODO(psyche): Unfortunately, this wrapper does not work correctly, due to a TS limitation.
   *
   * For a reason I do not understand, TS cannot resolve the parameter and return types of overloaded functions. It
   * only resolves one of the overload signatures.
   *
   * `addAppListener` has an overloaded type signature, so `Parameters<typeof addAppListener>[0]` resolves to only one
   * of the 5 possible arg types for the function. Unfortunately, you can't use this wrapper in the same way you could
   * if you called `addAppListener` directly.
   *
   * There are a number of proposed solutions but none worked for me. I think there may be limitations with the use of
   * generics? See:
   * - https://github.com/microsoft/TypeScript/issues/32164
   * - https://github.com/microsoft/TypeScript/issues/29732
   */
  addStoreListener = (arg: Parameters<typeof addAppListener>[0]) => {
    return this.store.dispatch(addAppListener(arg));
  };

  /**
   * Gets the canvas slice.
   *
   * The state is stored in redux.
   */
  getCanvasState = (): CanvasState => {
    return this.runSelector(selectCanvasSlice);
  };

  /**
   * Resets an entity, pushing state to redux.
   */
  resetEntity = (arg: EntityIdentifierPayload) => {
    this.store.dispatch(entityReset(arg));
  };

  /**
   * Updates an entity's position, pushing state to redux.
   */
  setEntityPosition = (arg: EntityMovedToPayload) => {
    this.store.dispatch(entityMovedTo(arg));
  };

  /**
   * Moves an entity by the give offset, pushing state to redux.
   */
  moveEntityBy = (arg: EntityMovedByPayload) => {
    this.store.dispatch(entityMovedBy(arg));
  };

  /**
   * Adds a brush line to an entity, pushing state to redux.
   */
  addBrushLine = (arg: EntityBrushLineAddedPayload) => {
    this.store.dispatch(entityBrushLineAdded(arg));
  };

  /**
   * Adds an eraser line to an entity, pushing state to redux.
   */
  addEraserLine = (arg: EntityEraserLineAddedPayload) => {
    this.store.dispatch(entityEraserLineAdded(arg));
  };

  /**
   * Adds a rectangle to an entity, pushing state to redux.
   */
  addRect = (arg: EntityRectAddedPayload) => {
    this.store.dispatch(entityRectAdded(arg));
  };

  /**
   * Adds a raster layer to the canvas, pushing state to redux.
   */
  addRasterLayer = (arg: Param0<typeof rasterLayerAdded>) => {
    this.store.dispatch(rasterLayerAdded(arg));
  };

  /**
   * Adds a control layer to the canvas, pushing state to redux.
   */
  addControlLayer = (arg: Param0<typeof controlLayerAdded>) => {
    this.store.dispatch(controlLayerAdded(arg));
  };

  /**
   * Adds an inpaint mask to the canvas, pushing state to redux.
   */
  addInpaintMask = (arg: Param0<typeof inpaintMaskAdded>) => {
    this.store.dispatch(inpaintMaskAdded(arg));
  };

  /**
   * Adds regional guidance to the canvas, pushing state to redux.
   */
  addRegionalGuidance = (arg: Param0<typeof rgAdded>) => {
    this.store.dispatch(rgAdded(arg));
  };

  /**
   * Rasterizes an entity, pushing state to redux.
   */
  rasterizeEntity = (arg: EntityRasterizedPayload) => {
    this.store.dispatch(entityRasterized(arg));
  };

  /**
   * Sets the generation bbox rect, pushing state to redux.
   */
  setGenerationBbox = (rect: Rect) => {
    this.store.dispatch(bboxChangedFromCanvas(rect));
  };

  /**
   * Sets the brush width, pushing state to redux.
   */
  setBrushWidth = (width: number) => {
    this.store.dispatch(settingsBrushWidthChanged(width));
  };

  /**
   * Sets the eraser width, pushing state to redux.
   */
  setEraserWidth = (width: number) => {
    this.store.dispatch(settingsEraserWidthChanged(width));
  };

  /**
   * Sets the drawing color, pushing state to redux.
   */
  setColor = (color: Partial<RgbaColor>) => {
    return this.getSettings().activeColor === 'bgColor'
      ? this.store.dispatch(settingsBgColorChanged(color))
      : this.store.dispatch(settingsFgColorChanged(color));
  };

  /**
   * Run a graph and return an image output. The specified output node must return an image output, else the promise
   * will reject with an error.
   *
   * @param arg The arguments for the function.
   * @param arg.graph The graph to execute.
   * @param arg.outputNodeId The id of the node whose output will be retrieved.
   * @param arg.destination The destination to assign to the batch. If omitted, the destination is not set.
   * @param arg.prepend Whether to prepend the graph to the front of the queue. If omitted, the graph is appended to the end of the queue.
   * @param arg.timeout The timeout for the batch. If omitted, there is no timeout.
   * @param arg.signal An optional signal to cancel the operation. If omitted, the operation cannot be canceled!
   *
   * @returns A promise that resolves to the image output or rejects with an error.
   *
   * @example
   *
   * ```ts
   * const graph = new Graph();
   * const outputNode = graph.addNode({ id: 'my-resize-node', type: 'img_resize', image: { image_name: 'my-image.png' } });
   * const controller = new AbortController();
   * const imageDTO = await this.manager.stateApi.runGraphAndReturnImageOutput({
   *  graph,
   *  outputNodeId: outputNode.id,
   *  prepend: true,
   *  signal: controller.signal,
   * });
   * // To cancel the operation:
   * controller.abort();
   * ```
   */
  runGraphAndReturnImageOutput = async (arg: {
    graph: Graph;
    outputNodeId: string;
    options?: RunGraphOptions;
  }): Promise<ImageDTO> => {
    const dependencies = buildRunGraphDependencies(this.store.dispatch, this.manager.socket);

    const { output } = await runGraph({
      dependencies,
      ...arg,
    });

    // Extract the image from the result - we expect a single image
    const imageDTO = await this.getImageDTOFromResult(output);

    return imageDTO;
  };

  /**
   * Helper function to extract ImageDTO from graph execution result.
   * Expects the result to be an ImageOutput.
   */
  private getImageDTOFromResult = async (result: S['GraphExecutionState']['results'][string]): Promise<ImageDTO> => {
    // Validate that the result is an ImageOutput using zod schema
    const parseResult = zImageOutput.safeParse(result);
    if (!parseResult.success) {
      throw new Error(`Graph output is not a valid ImageOutput. Got: ${JSON.stringify(result)}`);
    }

    const imageOutput = parseResult.data;
    return await getImageDTO(imageOutput.image.image_name);
  };

  /**
   * Gets the generation bbox state from redux.
   */
  getBbox = () => {
    return this.runSelector(selectBbox);
  };

  /**
   * Gets the canvas settings from redux.
   */
  getSettings = () => {
    return this.runSelector(selectCanvasSettingsSlice);
  };

  /**
   * Gets the _positional_ grid size for the current canvas. Note that this is not the same as bbox grid size, which is
   * based on the currently-selected model.
   */
  getPositionGridSize = (): number => {
    const snapToGrid = this.getSettings().snapToGrid;
    if (!snapToGrid) {
      const overrideSnap = this.$ctrlKey.get() || this.$metaKey.get();
      if (overrideSnap) {
        const useFine = this.$shiftKey.get();
        if (useFine) {
          return 8;
        }
        return 64;
      }
      return 1;
    }
    const useFine = this.$ctrlKey.get() || this.$metaKey.get();
    if (useFine) {
      return 8;
    }
    return 64;
  };

  /**
   * Gets the regions state from redux.
   */
  getRegionsState = () => {
    return this.getCanvasState().regionalGuidance;
  };

  /**
   * Gets the raster layers state from redux.
   */
  getRasterLayersState = () => {
    return this.getCanvasState().rasterLayers;
  };

  /**
   * Gets the control layers state from redux.
   */
  getControlLayersState = () => {
    return this.getCanvasState().controlLayers;
  };

  /**
   * Gets the inpaint masks state from redux.
   */
  getInpaintMasksState = () => {
    return this.getCanvasState().inpaintMasks;
  };

  /**
   * Gets the canvas staging area state from redux.
   */
  getStagingArea = () => {
    return this.runSelector(selectCanvasSessionSlice);
  };

  /**
   * Gets the grid size for the current canvas, based on the currently-selected model
   */
  getBboxGridSize = (): number => {
    return this.runSelector(selectGridSize);
  };

  /**
   * Checks if an entity is selected.
   */
  getIsSelected = (id: string): boolean => {
    return this.getCanvasState().selectedEntityIdentifier?.id === id;
  };

  /**
   * Gets the number of entities that are currently rendered on the canvas.
   */
  getRenderedEntityCount = (): number => {
    const renderableEntities = selectAllRenderableEntities(this.getCanvasState());
    let count = 0;
    for (const entity of renderableEntities) {
      if (entity.isEnabled) {
        count++;
      }
    }
    return count;
  };

  /**
   * Gets the currently selected entity's adapter
   */
  getSelectedEntityAdapter = (): CanvasEntityAdapter | null => {
    const state = this.getCanvasState();
    if (!state.selectedEntityIdentifier) {
      return null;
    }
    return this.manager.getAdapter(state.selectedEntityIdentifier);
  };

  /**
   * Gets the current drawing color.
   *
   * The color is determined by the tool state, except when the selected entity is a regional guidance or inpaint mask.
   * In that case, the color is always black.
   *
   * Regional guidance and inpaint mask entities use a compositing rect to draw with their selected color and texture,
   * so the color for lines and rects doesn't matter - it is never seen. The only requirement is that it is opaque. For
   * consistency with conventional black and white mask images, we use black as the color for these entities.
   */
  getCurrentColor = (): RgbaColor => {
    let color: RgbaColor =
      this.getSettings().activeColor === 'bgColor' ? this.getSettings().bgColor : this.getSettings().fgColor;
    const selectedEntity = this.getSelectedEntityAdapter();
    if (selectedEntity) {
      // These two entity types use a compositing rect for opacity. Their fill is always a solid color.
      if (selectedEntity.state.type === 'regional_guidance' || selectedEntity.state.type === 'inpaint_mask') {
        color = RGBA_BLACK;
      }
    }
    return color;
  };

  /**
   * Gets the brush preview color. The brush preview color is determined by the tool state and the selected entity.
   *
   * The color is the tool state's color, except when the selected entity is a regional guidance or inpaint mask.
   *
   * These entities have their own color and texture, so the brush preview should use those instead of the tool state's
   * color.
   */
  getBrushPreviewColor = (): RgbaColor => {
    const selectedEntity = this.getSelectedEntityAdapter();
    if (selectedEntity?.state.type === 'regional_guidance' || selectedEntity?.state.type === 'inpaint_mask') {
      // TODO(psyche): If we move the brush preview's Konva nodes to the selected entity renderer, we can draw them
      // under the entity's compositing rect, so they would use selected entity's selected color and texture. As a
      // temporary workaround to improve the UX when using a brush on a regional guidance or inpaint mask, we use the
      // selected entity's fill color with 50% opacity.
      return { ...selectedEntity.state.fill.color, a: 0.5 };
    } else {
      return this.getSettings().activeColor === 'bgColor' ? this.getSettings().bgColor : this.getSettings().fgColor;
    }
  };

  /**
   * The entity adapter being filtered, if any.
   */
  $filteringAdapter = atom<CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer | null>(null);

  /**
   * Whether an entity is currently being filtered. Derived from `$filteringAdapter`.
   */
  $isFiltering = computed(this.$filteringAdapter, (filteringAdapter) => Boolean(filteringAdapter));

  /**
   * The entity adapter being transformed, if any.
   */
  $transformingAdapter = atom<CanvasEntityAdapter | null>(null);

  /**
   * Whether an entity is currently being transformed. Derived from `$transformingAdapter`.
   */
  $isTransforming = computed(this.$transformingAdapter, (transformingAdapter) => Boolean(transformingAdapter));

  /**
   * The entity adapter being rasterized, if any.
   */
  $rasterizingAdapter = atom<CanvasEntityAdapter | null>(null);

  /**
   * Whether an entity is currently being rasterized. Derived from `$rasterizingAdapter`.
   */
  $isRasterizing = computed(this.$rasterizingAdapter, (rasterizingAdapter) => Boolean(rasterizingAdapter));

  /**
   * The entity adapter being segmented, if any.
   */
  $segmentingAdapter = atom<CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer | null>(null);

  /**
   * Whether an entity is currently being segmented. Derived from `$segmentingAdapter`.
   */
  $isSegmenting = computed(this.$segmentingAdapter, (segmentingAdapter) => Boolean(segmentingAdapter));

  /**
   * Whether the space key is currently pressed.
   */
  $spaceKey = atom<boolean>(false);

  /**
   * Whether the alt key is currently pressed.
   */
  $altKey = $alt;

  /**
   * Whether the ctrl key is currently pressed.
   */
  $ctrlKey = $ctrl;

  /**
   * Whether the meta key is currently pressed.
   */
  $metaKey = $meta;

  /**
   * Whether the shift key is currently pressed.
   */
  $shiftKey = $shift;

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      $filteringAdapter: this.$filteringAdapter.get()?.entityIdentifier ?? null,
      $isFiltering: this.$isFiltering.get(),
      $transformingAdapter: this.$transformingAdapter.get()?.entityIdentifier ?? null,
      $isTransforming: this.$isTransforming.get(),
      $rasterizingAdapter: this.$rasterizingAdapter.get()?.entityIdentifier ?? null,
      $isRasterizing: this.$isRasterizing.get(),
      $segmentingAdapter: this.$segmentingAdapter.get()?.entityIdentifier ?? null,
      $isSegmenting: this.$isSegmenting.get(),
    };
  };
}
