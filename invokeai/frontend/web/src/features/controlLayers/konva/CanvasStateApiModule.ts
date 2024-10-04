import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { Selector } from '@reduxjs/toolkit';
import { addAppListener } from 'app/store/middleware/listenerMiddleware';
import type { AppStore, RootState } from 'app/store/store';
import { withResultAsync } from 'common/util/result';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { SubscriptionHandler } from 'features/controlLayers/konva/util';
import { createReduxSubscription, getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectCanvasSettingsSlice,
  settingsBrushWidthChanged,
  settingsColorChanged,
  settingsEraserWidthChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import {
  bboxChangedFromCanvas,
  entityBrushLineAdded,
  entityEraserLineAdded,
  entityMoved,
  entityRasterized,
  entityRectAdded,
  entityReset,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasStagingAreaSlice } from 'features/controlLayers/store/canvasStagingAreaSlice';
import {
  selectAllRenderableEntities,
  selectBbox,
  selectCanvasSlice,
  selectGridSize,
} from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityType,
  CanvasState,
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
  Rect,
  RgbaColor,
} from 'features/controlLayers/store/types';
import { RGBA_BLACK } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { getImageDTO } from 'services/api/endpoints/images';
import { queueApi } from 'services/api/endpoints/queue';
import type { BatchConfig, ImageDTO, S } from 'services/api/types';
import { QueueError } from 'services/events/errors';
import { assert } from 'tsafe';

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
  setEntityPosition = (arg: EntityMovedPayload) => {
    this.store.dispatch(entityMoved(arg));
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
  setColor = (color: RgbaColor) => {
    return this.store.dispatch(settingsColorChanged(color));
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
  runGraphAndReturnImageOutput = (arg: {
    graph: Graph;
    outputNodeId: string;
    destination?: string;
    prepend?: boolean;
    timeout?: number;
    signal?: AbortSignal;
  }): Promise<ImageDTO> => {
    const { graph, outputNodeId, destination, prepend, timeout, signal } = arg;

    if (!graph.hasNode(outputNodeId)) {
      throw new Error(`Graph does not contain node with id: ${outputNodeId}`);
    }

    /**
     * We will use the origin to handle events from the graph. Ideally we'd just use the queue item's id, but there's a
     * race condition:
     * - The queue item id is not available until the graph is enqueued
     * - The graph may complete before we can set up the listeners to handle the completion event
     *
     * The origin is the only unique identifier we have that is guaranteed to be available before the graph is enqueued,
     * so we will use that to filter events.
     */
    const origin = getPrefixedId(graph.id);

    const batch: BatchConfig = {
      prepend,
      batch: {
        graph: graph.getGraph(),
        origin,
        destination,
        runs: 1,
      },
    };

    /**
     * If a timeout is provided, we will cancel the graph if it takes too long - but we need a way to clear the timeout
     * if the graph completes or errors before the timeout.
     */
    let timeoutId: number | null = null;
    const _clearTimeout = () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
        timeoutId = null;
      }
    };

    // There's a bit of a catch-22 here: we need to set the cancelGraph callback before we enqueue the graph, but we
    // can't set it until we have the batch_id from the enqueue request. So we'll set a dummy function here and update
    // it later.
    let cancelGraph: () => void = () => {
      this.log.warn('cancelGraph called before cancelGraph is set');
    };

    const resultPromise = new Promise<ImageDTO>((resolve, reject) => {
      const invocationCompleteHandler = async (event: S['InvocationCompleteEvent']) => {
        // Ignore events that are not for this graph
        if (event.origin !== origin) {
          return;
        }

        // Ignore events that are not from the output node
        if (event.invocation_source_id !== outputNodeId) {
          return;
        }

        // If we get here, the event is for the correct graph and output node.

        // Clear the timeout and socket listeners
        _clearTimeout();
        clearListeners();

        // The result must be an image output
        const { result } = event;
        if (result.type !== 'image_output') {
          reject(new Error(`Graph output node did not return an image output, got: ${result}`));
          return;
        }

        // Get the result image DTO
        const getImageDTOResult = await withResultAsync(() => getImageDTO(result.image.image_name));
        if (getImageDTOResult.isErr()) {
          reject(getImageDTOResult.error);
          return;
        }

        // Ok!
        resolve(getImageDTOResult.value);
      };

      const queueItemStatusChangedHandler = (event: S['QueueItemStatusChangedEvent']) => {
        // Ignore events that are not for this graph
        if (event.origin !== origin) {
          return;
        }

        // Ignore events where the status is pending or in progress - no need to do anything for these
        if (event.status === 'pending' || event.status === 'in_progress') {
          return;
        }

        if (event.status === 'completed') {
          /**
           * The invocation_complete event should have been received before the queue item completed event, and the
           * event listeners are cleared in the invocation_complete handler. If we get here, it means we never got
           * the completion event for the output node! This should is a fail case.
           *
           * TODO(psyche): In the unexpected case where events are received out of order, this logic doesn't do what
           * we expect. If we got a queue item completed event before the output node completion event, we'd erroneously
           * triggers this error.
           *
           * For now, we'll just log a warning instead of rejecting the promise. This should be super rare anyways.
           */
          // reject(new Error('Queue item completed without output node completion event'));
          this.log.warn('Queue item completed without output node completion event');
          return;
        }

        // event.status is 'failed', 'canceled' - something has gone awry
        _clearTimeout();
        clearListeners();

        if (event.status === 'failed') {
          // We expect the event to have error details, but technically it's possible that it doesn't
          const { error_type, error_message, error_traceback } = event;
          if (error_type && error_message && error_traceback) {
            reject(new QueueError(error_type, error_message, error_traceback));
          } else {
            reject(new Error('Queue item failed, but no error details were provided'));
          }
        } else {
          // event.status is 'canceled'
          reject(new Error('Graph canceled'));
        }
      };

      // We are ready to enqueue the graph
      const enqueueRequest = this.store.dispatch(
        queueApi.endpoints.enqueueBatch.initiate(batch, {
          // Use the same cache key for all enqueueBatch requests, so that all consumers of this query get the same status
          // updates.
          fixedCacheKey: 'enqueueBatch',
          // We do not need RTK to track this request in the store
          track: false,
        })
      );

      // Enqueue the graph and get the batch_id, updating the cancel graph callack. We need to do this in a .then() block
      // instead of awaiting the promise to avoid await-ing in a promise executor. Also need to catch any errors.
      enqueueRequest
        .unwrap()
        .then((data) => {
          // The `batch_id` should _always_ be present - the OpenAPI schema from which the types are generated is incorrect.
          // TODO(psyche): Fix the OpenAPI schema.
          const batch_id = data.batch.batch_id;
          assert(batch_id, 'Enqueue result is missing batch_id');
          cancelGraph = () => {
            this.store.dispatch(
              queueApi.endpoints.cancelByBatchIds.initiate({ batch_ids: [batch_id] }, { track: false })
            );
          };
        })
        .catch((error) => {
          reject(error);
        });

      this.manager.socket.on('invocation_complete', invocationCompleteHandler);
      this.manager.socket.on('queue_item_status_changed', queueItemStatusChangedHandler);

      const clearListeners = () => {
        this.manager.socket.off('invocation_complete', invocationCompleteHandler);
        this.manager.socket.off('queue_item_status_changed', queueItemStatusChangedHandler);
      };

      if (timeout) {
        timeoutId = window.setTimeout(() => {
          this.log.trace('Graph canceled by timeout');
          clearListeners();
          cancelGraph();
          reject(new Error('Graph timed out'));
        }, timeout);
      }

      if (signal) {
        signal.addEventListener('abort', () => {
          this.log.trace('Graph canceled by signal');
          _clearTimeout();
          clearListeners();
          cancelGraph();
          reject(new Error('Graph canceled'));
        });
      }
    });

    return resultPromise;
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
  getGridSize = (): number => {
    const snapToGrid = this.getSettings().snapToGrid;
    if (!snapToGrid) {
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
    return this.runSelector(selectCanvasStagingAreaSlice);
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
   * Checks if an entity type is hidden. Individual entities are not hidden; the entire entity type is hidden.
   */
  getIsTypeHidden = (type: CanvasEntityType): boolean => {
    switch (type) {
      case 'raster_layer':
        return this.getRasterLayersState().isHidden;
      case 'control_layer':
        return this.getControlLayersState().isHidden;
      case 'inpaint_mask':
        return this.getInpaintMasksState().isHidden;
      case 'regional_guidance':
        return this.getRegionsState().isHidden;
      default:
        assert(false, 'Unhandled entity type');
    }
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
    if (state.selectedEntityIdentifier) {
      return this.manager.getAdapter(state.selectedEntityIdentifier);
    }
    return null;
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
    let color: RgbaColor = this.getSettings().color;
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
      return this.getSettings().color;
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
   * Whether an entity is currently being transformed. Derived from `$transformingAdapter`.
   */
  $isRasterizing = computed(this.$rasterizingAdapter, (rasterizingAdapter) => Boolean(rasterizingAdapter));

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
}
