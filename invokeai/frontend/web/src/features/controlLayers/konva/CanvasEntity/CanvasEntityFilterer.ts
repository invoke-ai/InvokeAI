import type { SerializableObject } from 'common/types';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectAutoPreviewFilter } from 'features/controlLayers/store/canvasSettingsSlice';
import type { CanvasImageState, FilterConfig } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS, imageDTOToImageObject } from 'features/controlLayers/store/types';
import { debounce } from 'lodash-es';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { getImageDTO } from 'services/api/endpoints/images';
import type { BatchConfig, ImageDTO, S } from 'services/api/types';
import { assert } from 'tsafe';

export class CanvasEntityFilterer extends CanvasModuleBase {
  readonly type = 'canvas_filterer';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  imageState: CanvasImageState | null = null;
  subscriptions = new Set<() => void>();

  $isFiltering = atom<boolean>(false);
  $isProcessing = atom<boolean>(false);
  $filterConfig = atom<FilterConfig>(IMAGE_FILTERS.canny_image_processor.buildDefaults());

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
        if (this.manager.stateApi.getSettings().autoPreviewFilter && this.$isFiltering.get()) {
          this.previewFilter();
        }
      })
    );
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectAutoPreviewFilter, (autoPreviewFilter) => {
        if (autoPreviewFilter && this.$isFiltering.get()) {
          this.previewFilter();
        }
      })
    );
  }

  startFilter = (config?: FilterConfig) => {
    this.log.trace('Initializing filter');
    if (config) {
      this.$filterConfig.set(config);
    }
    this.$isFiltering.set(true);
    this.manager.stateApi.$filteringAdapter.set(this.parent);
    this.previewFilter();
  };

  previewFilter = debounce(
    async () => {
      const config = this.$filterConfig.get();
      const isValid = IMAGE_FILTERS[config.type].validateConfig?.(config as never) ?? true;
      if (!isValid) {
        return;
      }

      this.log.trace({ config }, 'Previewing filter');
      const rect = this.parent.transformer.getRelativeRect();
      const imageDTO = await this.parent.renderer.rasterize({ rect, attrs: { filters: [] } });
      const nodeId = getPrefixedId('filter_node');
      const batch = this.buildBatchConfig(imageDTO, config, nodeId);

      // Listen for the filter processing completion event
      const listener = async (event: S['InvocationCompleteEvent']) => {
        if (event.origin !== this.id || event.invocation_source_id !== nodeId) {
          return;
        }
        this.manager.socket.off('invocation_complete', listener);

        this.log.trace({ event } as SerializableObject, 'Handling filter processing completion');

        const { result } = event;
        assert(result.type === 'image_output', `Processor did not return an image output, got: ${result}`);

        const imageDTO = await getImageDTO(result.image.image_name);
        assert(imageDTO, "Failed to fetch processor output's image DTO");

        this.imageState = imageDTOToImageObject(imageDTO);

        await this.parent.bufferRenderer.setBuffer(this.imageState, true);

        this.parent.renderer.hideObjects();
        this.$isProcessing.set(false);
      };

      this.manager.socket.on('invocation_complete', listener);

      this.log.trace({ batch } as SerializableObject, 'Enqueuing filter batch');

      this.$isProcessing.set(true);
      this.manager.stateApi.enqueueBatch(batch);
    },
    1000,
    { leading: true, trailing: true }
  );

  applyFilter = () => {
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
      rect: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: imageState.image.height,
        height: imageState.image.width,
      },
      replaceObjects: true,
    });
    this.imageState = null;
    this.$isFiltering.set(false);
    this.manager.stateApi.$filteringAdapter.set(null);
  };

  cancelFilter = () => {
    this.log.trace('Cancelling filter');

    this.parent.bufferRenderer.clearBuffer();
    this.parent.renderer.showObjects();
    this.parent.transformer.updatePosition();
    this.parent.renderer.syncCache(true);
    this.imageState = null;
    this.$isProcessing.set(false);
    this.$isFiltering.set(false);
    this.manager.stateApi.$filteringAdapter.set(null);
  };

  buildBatchConfig = (imageDTO: ImageDTO, config: FilterConfig, id: string): BatchConfig => {
    // TODO(psyche): I can't get TS to be happy, it thinkgs `config` is `never` but it should be inferred from the generic... I'll just cast it for now
    const node = IMAGE_FILTERS[config.type].buildNode(imageDTO, config as never);
    node.id = id;
    const batch: BatchConfig = {
      prepend: true,
      batch: {
        graph: {
          nodes: {
            [node.id]: {
              ...node,
              // filtered images are always intermediate - do not save to gallery
              is_intermediate: true,
            },
          },
          edges: [],
        },
        origin: this.id,
        runs: 1,
      },
    };

    return batch;
  };
}
