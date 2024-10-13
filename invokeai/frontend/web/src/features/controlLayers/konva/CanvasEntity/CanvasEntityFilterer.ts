import { withResult, withResultAsync } from 'common/util/result';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectAutoProcessFilter } from 'features/controlLayers/store/canvasSettingsSlice';
import type { FilterConfig } from 'features/controlLayers/store/filters';
import { getFilterForModel, IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import type { CanvasImageState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { debounce } from 'lodash-es';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';
import { buildSelectModelConfig } from 'services/api/hooks/modelsByType';
import { isControlNetOrT2IAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

type CanvasEntityFiltererConfig = {
  processDebounceMs: number;
};

const DEFAULT_CONFIG: CanvasEntityFiltererConfig = {
  processDebounceMs: 1000,
};

export class CanvasEntityFilterer extends CanvasModuleBase {
  readonly type = 'canvas_filterer';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  imageState: CanvasImageState | null = null;
  subscriptions = new Set<() => void>();
  config: CanvasEntityFiltererConfig = DEFAULT_CONFIG;

  /**
   * The AbortController used to cancel the filter processing.
   */
  abortController: AbortController | null = null;

  $isFiltering = atom<boolean>(false);
  $hasProcessed = atom<boolean>(false);
  $isProcessing = atom<boolean>(false);
  $filterConfig = atom<FilterConfig>(IMAGE_FILTERS.canny_edge_detection.buildDefaults());

  constructor(parent: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = this.parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating filter module');

    this.subscriptions.add(
      this.$filterConfig.listen(() => {
        if (this.manager.stateApi.getSettings().autoProcessFilter && this.$isFiltering.get()) {
          this.process();
        }
      })
    );
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectAutoProcessFilter, (autoPreviewFilter) => {
        if (autoPreviewFilter && this.$isFiltering.get()) {
          this.process();
        }
      })
    );
  }

  start = (config?: FilterConfig) => {
    const filteringAdapter = this.manager.stateApi.$filteringAdapter.get();
    if (filteringAdapter) {
      assert(false, `Already filtering an entity: ${filteringAdapter.id}`);
    }

    this.log.trace('Initializing filter');
    if (config) {
      this.$filterConfig.set(config);
    } else if (this.parent.type === 'control_layer_adapter' && this.parent.state.controlAdapter.model) {
      // If the parent is a control layer adapter, we should check if the model has a default filter and set it if so
      const selectModelConfig = buildSelectModelConfig(
        this.parent.state.controlAdapter.model.key,
        isControlNetOrT2IAdapterModelConfig
      );
      const modelConfig = this.manager.stateApi.runSelector(selectModelConfig);
      const filter = getFilterForModel(modelConfig);
      this.$filterConfig.set(filter.buildDefaults());
    } else {
      // Otherwise, set the default filter
      this.$filterConfig.set(IMAGE_FILTERS.canny_edge_detection.buildDefaults());
    }
    this.$isFiltering.set(true);
    this.manager.stateApi.$filteringAdapter.set(this.parent);
    if (this.manager.stateApi.getSettings().autoProcessFilter) {
      this.processImmediate();
    }
  };

  processImmediate = async () => {
    const config = this.$filterConfig.get();
    const filterData = IMAGE_FILTERS[config.type];

    // Cannot get TS to be happy with `config`, thinks it should be `never`... eh...
    const isValid = filterData.validateConfig?.(config as never) ?? true;
    if (!isValid) {
      this.log.error({ config }, 'Invalid filter config');
      return;
    }

    this.log.trace({ config }, 'Processing filter');
    const rect = this.parent.transformer.getRelativeRect();

    const rasterizeResult = await withResultAsync(() =>
      this.parent.renderer.rasterize({ rect, attrs: { filters: [], opacity: 1 } })
    );
    if (rasterizeResult.isErr()) {
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
        // The filter graph should always be prepended to the queue so it's processed ASAP.
        prepend: true,
        /**
         * The filter node may need to download a large model. Currently, the models required by the filter nodes are
         * downloaded just-in-time, as required by the filter. If we use a timeout here, we might get into a catch-22
         * where the filter node is waiting for the model to download, but the download gets canceled if the filter
         * node times out.
         *
         * (I suspect the model download will actually _not_ be canceled if the graph is canceled, but let's not chance it!)
         *
         * TODO(psyche): Figure out a better way to handle this. Probably need to download the models ahead of time.
         */
        // timeout: 5000,
        /**
         * The filter node should be able to cancel the request if it's taking too long. This will cancel the graph's
         * queue item and clear any event listeners on the request.
         */
        signal: controller.signal,
      })
    );
    if (filterResult.isErr()) {
      this.log.error({ error: serializeError(filterResult.error) }, 'Error processing filter');
      this.$isProcessing.set(false);
      this.abortController = null;
      return;
    }

    this.log.trace({ imageDTO: filterResult.value }, 'Filter processed');
    this.imageState = imageDTOToImageObject(filterResult.value);

    await this.parent.bufferRenderer.setBuffer(this.imageState, true);

    this.$isProcessing.set(false);
    this.$hasProcessed.set(true);
    this.abortController = null;
  };

  process = debounce(this.processImmediate, this.config.processDebounceMs);

  apply = () => {
    const imageState = this.imageState;
    if (!imageState) {
      this.log.warn('No image state to apply filter to');
      return;
    }
    this.log.trace('Applying filter');
    this.parent.bufferRenderer.commitBuffer();
    const rect = this.parent.transformer.getRelativeRect();
    this.manager.stateApi.rasterizeEntity({
      entityIdentifier: this.parent.entityIdentifier,
      imageObject: imageState,
      position: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
      },
      replaceObjects: true,
    });
    this.imageState = null;
    this.$isFiltering.set(false);
    this.$hasProcessed.set(false);
    this.manager.stateApi.$filteringAdapter.set(null);
  };

  reset = () => {
    this.log.trace('Resetting filter');

    this.abortController?.abort();
    this.abortController = null;
    this.parent.bufferRenderer.clearBuffer();
    this.parent.transformer.updatePosition();
    this.parent.renderer.syncKonvaCache(true);
    this.imageState = null;
    this.$hasProcessed.set(false);
  };

  cancel = () => {
    this.log.trace('Cancelling filter');

    this.reset();
    this.$isProcessing.set(false);
    this.$isFiltering.set(false);
    this.$hasProcessed.set(false);
    this.manager.stateApi.$filteringAdapter.set(null);
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      config: this.config,
      $isFiltering: this.$isFiltering.get(),
      $hasProcessed: this.$hasProcessed.get(),
      $isProcessing: this.$isProcessing.get(),
      $filterConfig: this.$filterConfig.get(),
    };
  };
}
