import { deepClone } from 'common/util/deepClone';
import { withResult, withResultAsync } from 'common/util/result';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { CanvasObjectImage } from 'features/controlLayers/konva/CanvasObject/CanvasObjectImage';
import { addCoords, getKonvaNodeDebugAttrs, getPrefixedId } from 'features/controlLayers/konva/util';
import { selectAutoProcess } from 'features/controlLayers/store/canvasSettingsSlice';
import type { FilterConfig } from 'features/controlLayers/store/filters';
import { getFilterForModel, IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import type { CanvasImageState, CanvasRenderableEntityType } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { toast } from 'features/toast/toast';
import Konva from 'konva';
import { debounce } from 'lodash-es';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';
import { buildSelectModelConfig } from 'services/api/hooks/modelsByType';
import { isControlLayerModelConfig } from 'services/api/types';
import stableHash from 'stable-hash';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

type CanvasEntityFiltererConfig = {
  /**
   * The debounce time in milliseconds for processing the filter.
   */
  PROCESS_DEBOUNCE_MS: number;
};

const DEFAULT_CONFIG: CanvasEntityFiltererConfig = {
  PROCESS_DEBOUNCE_MS: 1000,
};

export class CanvasEntityFilterer extends CanvasModuleBase {
  readonly type = 'canvas_filterer';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  config: CanvasEntityFiltererConfig = DEFAULT_CONFIG;

  subscriptions = new Set<() => void>();

  /**
   * The AbortController used to cancel the filter processing.
   */
  abortController: AbortController | null = null;

  /**
   * Whether the module is currently filtering an image.
   */
  $isFiltering = atom<boolean>(false);
  /**
   * The hash of the last processed config. This is used to prevent re-processing the same config.
   */
  $lastProcessedHash = atom<string>('');

  /**
   * Whether the module is currently processing the filter.
   */
  $isProcessing = atom<boolean>(false);

  /**
   * The config for the filter.
   */
  $filterConfig = atom<FilterConfig>(IMAGE_FILTERS.adjust_image.buildDefaults());

  /**
   * The initial filter config, used to reset the filter config.
   */
  $initialFilterConfig = atom<FilterConfig | null>(null);

  /**
   * The ephemeral image state of the filtered image.
   */
  $imageState = atom<CanvasImageState | null>(null);

  /**
   * Whether the module has an image state. This is a computed value based on $imageState.
   */
  $hasImageState = computed(this.$imageState, (imageState) => imageState !== null);

  /**
   * Whether the filter is in simple mode. In simple mode, the filter is started with a default filter config and the
   * user is not presented with filter settings.
   */
  $simple = atom<boolean>(false);

  /**
   * The filtered image object module, if it exists.
   */
  imageModule: CanvasObjectImage | null = null;

  /**
   * The Konva nodes for the module.
   */
  konva: {
    /**
     * The main Konva group node for the module. This is added to the parent layer on start, and removed on teardown.
     */
    group: Konva.Group;
  };

  KONVA_GROUP_NAME = `${this.type}:group`;

  constructor(parent: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating filter module');

    this.konva = {
      group: new Konva.Group({ name: this.KONVA_GROUP_NAME }),
    };
  }

  /**
   * Adds event listeners needed while filtering the entity.
   */
  subscribe = () => {
    // As the filter config changes, process the filter
    this.subscriptions.add(
      this.$filterConfig.listen(() => {
        if (this.manager.stateApi.getSettings().autoProcess && this.$isFiltering.get()) {
          this.process();
        }
      })
    );
    // When auto-process is enabled, process the filter
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectAutoProcess, (autoProcess) => {
        if (autoProcess && this.$isFiltering.get()) {
          this.process();
        }
      })
    );
  };

  /**
   * Removes event listeners used while filtering the entity.
   */
  unsubscribe = () => {
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
  };

  /**
   * Starts the filter module.
   * @param config The filter config to use. If omitted, the default filter config is used.
   */
  start = (config?: FilterConfig) => {
    const filteringAdapter = this.manager.stateApi.$filteringAdapter.get();
    if (filteringAdapter) {
      this.log.error(`Already filtering an entity: ${filteringAdapter.id}`);
      return;
    }

    this.log.trace('Initializing filter');

    // Reset any previous state
    this.resetEphemeralState();
    this.$isFiltering.set(true);

    // Update the konva group's position to match the parent entity
    const pixelRect = this.parent.transformer.$pixelRect.get();
    const position = addCoords(this.parent.state.position, pixelRect);
    this.konva.group.setAttrs(position);

    // Add the group to the parent layer
    this.parent.konva.layer.add(this.konva.group);

    if (config) {
      // If a config is provided, use it
      this.$filterConfig.set(config);
      this.$initialFilterConfig.set(config);
      this.$simple.set(true);
    } else {
      const initialConfig = this.createInitialFilterConfig();
      this.$filterConfig.set(initialConfig);
      this.$initialFilterConfig.set(initialConfig);
      this.$simple.set(false);
    }

    this.subscribe();

    this.manager.stateApi.$filteringAdapter.set(this.parent);

    if (this.manager.stateApi.getSettings().autoProcess) {
      this.processImmediate();
    }
  };

  createInitialFilterConfig = (): FilterConfig => {
    if (this.parent.type === 'control_layer_adapter' && this.parent.state.controlAdapter.model) {
      // If the parent is a control layer adapter, we should check if the model has a default filter and set it if so
      const selectModelConfig = buildSelectModelConfig(
        this.parent.state.controlAdapter.model.key,
        isControlLayerModelConfig
      );
      const modelConfig = this.manager.stateApi.runSelector(selectModelConfig);
      // This always returns a filter
      const filter = getFilterForModel(modelConfig) ?? IMAGE_FILTERS.canny_edge_detection;
      return filter.buildDefaults();
    } else {
      // Otherwise, used the default filter
      return IMAGE_FILTERS.adjust_image.buildDefaults();
    }
  };

  /**
   * Processes the filter, updating the module's state and rendering the filtered image.
   */
  processImmediate = async () => {
    if (!this.$isFiltering.get()) {
      this.log.warn('Cannot process filter when not initialized');
      return;
    }
    const config = this.$filterConfig.get();
    const filterData = IMAGE_FILTERS[config.type];

    // Cannot get TS to be happy with `config`, thinks it should be `never`... eh...
    const isValid = filterData.validateConfig?.(config as never) ?? true;
    if (!isValid) {
      this.log.error({ config }, 'Invalid filter config');
      return;
    }

    const hash = stableHash({ config });
    if (hash === this.$lastProcessedHash.get()) {
      this.log.trace('Already processed config');
      return;
    }

    this.log.trace({ config }, 'Processing filter');
    const rect = this.parent.transformer.getRelativeRect();

    const rasterizeResult = await withResultAsync(() =>
      this.parent.renderer.rasterize({ rect, attrs: { filters: [], opacity: 1 } })
    );
    if (rasterizeResult.isErr()) {
      toast({ status: 'error', title: 'Failed to process filter' });
      this.log.error({ error: serializeError(rasterizeResult.error) }, 'Error rasterizing entity');
      this.$isProcessing.set(false);
      return;
    }

    this.$isProcessing.set(true);

    const imageDTO = rasterizeResult.value;

    // Cannot get TS to be happy with `config`, thinks it should be `never`... eh...
    const buildGraphResult = withResult(() => filterData.buildGraph(imageDTO, config as never));
    if (buildGraphResult.isErr()) {
      this.log.error({ error: serializeError(buildGraphResult.error) }, 'Error building filter graph');
      this.$isProcessing.set(false);
      return;
    }

    const controller = new AbortController();
    this.abortController = controller;

    const { graph, outputNodeId } = buildGraphResult.value;

    const filterResult = await withResultAsync(() =>
      this.manager.stateApi.runGraphAndReturnImageOutput({
        graph,
        outputNodeId,
        prepend: true,
        signal: controller.signal,
      })
    );

    // If there is an error, log it and bail out of this processing run
    if (filterResult.isErr()) {
      this.log.error({ error: serializeError(filterResult.error) }, 'Error filtering');
      this.$isProcessing.set(false);
      // Clean up the abort controller as needed
      if (!controller.signal.aborted) {
        controller.abort();
      }
      this.abortController = null;
      return;
    }

    this.log.trace({ imageDTO: filterResult.value }, 'Filtered');

    // Prepare the ephemeral image state
    const imageState = imageDTOToImageObject(filterResult.value);
    this.$imageState.set(imageState);

    // Stash the existing image module - we will destroy it after the new image is rendered to prevent a flash
    // of an empty layer
    const oldImageModule = this.imageModule;

    this.imageModule = new CanvasObjectImage(imageState, this);

    // Force update the masked image - after awaiting, the image will be rendered (in memory)
    await this.imageModule.update(imageState, true);

    this.konva.group.add(this.imageModule.konva.group);

    // The filtered image have some transparency, so we need to hide the objects of the parent entity to prevent the
    // two images from blending. We will show the objects again in the teardown method, which is always called after
    // the filter finishes (applied or canceled).
    this.parent.renderer.hideObjects();

    if (oldImageModule) {
      // Destroy the old image module now that the new one is rendered
      oldImageModule.destroy();
    }

    // The porcessing is complete, set can set the last processed hash and isProcessing to false
    this.$lastProcessedHash.set(hash);

    this.$isProcessing.set(false);

    // Clean up the abort controller as needed
    if (!controller.signal.aborted) {
      controller.abort();
    }

    this.abortController = null;
  };

  /**
   * Debounced version of processImmediate.
   */
  process = debounce(this.processImmediate, this.config.PROCESS_DEBOUNCE_MS);

  /**
   * Applies the filter image to the entity, replacing the entity's objects with the filtered image.
   */
  apply = () => {
    const filteredImageObjectState = this.$imageState.get();
    if (!filteredImageObjectState) {
      this.log.warn('No image state to apply filter to');
      return;
    }
    this.log.trace('Applying');

    // Have the parent adopt the image module - this prevents a flash of the original layer content before the filtered
    // image is rendered
    if (this.imageModule) {
      this.parent.renderer.adoptObjectRenderer(this.imageModule);
    }

    // Rasterize the entity, replacing the objects with the masked image
    const rect = this.parent.transformer.getRelativeRect();
    this.manager.stateApi.rasterizeEntity({
      entityIdentifier: this.parent.entityIdentifier,
      imageObject: filteredImageObjectState,
      position: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
      },
      replaceObjects: true,
    });

    // Final cleanup and teardown, returning user to main canvas UI
    this.teardown();
  };

  /**
   * Saves the filtered image as a new entity of the given type.
   * @param type The type of entity to save the filtered image as.
   */
  saveAs = (type: CanvasRenderableEntityType) => {
    const imageState = this.$imageState.get();
    if (!imageState) {
      this.log.warn('No image state to apply filter to');
      return;
    }
    this.log.trace(`Saving as ${type}`);

    const rect = this.parent.transformer.getRelativeRect();
    const arg = {
      overrides: {
        objects: [imageState],
        position: {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
        },
      },
      isSelected: true,
    };

    switch (type) {
      case 'raster_layer':
        this.manager.stateApi.addRasterLayer(arg);
        break;
      case 'control_layer':
        this.manager.stateApi.addControlLayer(arg);
        break;
      case 'inpaint_mask':
        this.manager.stateApi.addInpaintMask(arg);
        break;
      case 'regional_guidance':
        this.manager.stateApi.addRegionalGuidance(arg);
        break;
      default:
        assert<Equals<typeof type, never>>(false);
    }
  };

  resetEphemeralState = () => {
    // First we need to bail out of any processing
    if (this.abortController && !this.abortController.signal.aborted) {
      this.abortController.abort();
    }
    this.abortController = null;

    // If the image module exists, and is a child of the group, destroy it. It might not be a child of the group if
    // the user has applied the filter and the image has been adopted by the parent entity.
    if (this.imageModule && this.imageModule.konva.group.parent === this.konva.group) {
      this.imageModule.destroy();
      this.imageModule = null;
    }
    const initialFilterConfig = deepClone(this.$initialFilterConfig.get() ?? this.createInitialFilterConfig());
    this.$filterConfig.set(initialFilterConfig);
    this.$imageState.set(null);
    this.$lastProcessedHash.set('');
    this.$isProcessing.set(false);
  };

  teardown = () => {
    this.unsubscribe();
    // Re-enable the objects of the parent entity
    this.parent.renderer.showObjects();
    this.konva.group.remove();
    // The reset must be done _after_ unsubscribing from listeners, in case the listeners would otherwise react to
    // the reset. For example, if auto-processing is enabled and we reset the state, it may trigger processing.
    this.resetEphemeralState();
    this.$isFiltering.set(false);
    this.manager.stateApi.$filteringAdapter.set(null);
  };

  /**
   * Resets the module (e.g. remove all points and the mask image).
   *
   * Does not cancel or otherwise complete the segmenting process.
   */
  reset = () => {
    this.log.trace('Resetting');
    this.resetEphemeralState();
  };

  cancel = () => {
    this.log.trace('Canceling');
    this.teardown();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      config: this.config,
      imageState: deepClone(this.$imageState.get()),
      $isFiltering: this.$isFiltering.get(),
      $lastProcessedHash: this.$lastProcessedHash.get(),
      $isProcessing: this.$isProcessing.get(),
      $filterConfig: this.$filterConfig.get(),
      konva: { group: getKonvaNodeDebugAttrs(this.konva.group) },
    };
  };

  destroy = () => {
    this.log.debug('Destroying module');
    if (this.abortController && !this.abortController.signal.aborted) {
      this.abortController.abort();
    }
    this.abortController = null;
    this.unsubscribe();
    this.konva.group.destroy();
  };
}
