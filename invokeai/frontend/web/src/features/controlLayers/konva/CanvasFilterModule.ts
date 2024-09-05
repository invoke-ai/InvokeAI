import type { SerializableObject } from 'common/types';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntityAdapter/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterInpaintMask } from 'features/controlLayers/konva/CanvasEntityAdapter/CanvasEntityAdapterInpaintMask';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntityAdapter/CanvasEntityAdapterRasterLayer';
import type { CanvasEntityAdapterRegionalGuidance } from 'features/controlLayers/konva/CanvasEntityAdapter/CanvasEntityAdapterRegionalGuidance';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasEntityIdentifier, CanvasImageState, FilterConfig } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS, imageDTOToImageObject } from 'features/controlLayers/store/types';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { getImageDTO } from 'services/api/endpoints/images';
import type { BatchConfig, ImageDTO, S } from 'services/api/types';
import { assert } from 'tsafe';

export class CanvasFilterModule extends CanvasModuleBase {
  readonly type = 'canvas_filter';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  imageState: CanvasImageState | null = null;

  $adapter = atom<
    | CanvasEntityAdapterRasterLayer
    | CanvasEntityAdapterControlLayer
    | CanvasEntityAdapterInpaintMask
    | CanvasEntityAdapterRegionalGuidance
    | null
  >(null);
  $isFiltering = computed(this.$adapter, (adapter) => Boolean(adapter));
  $isProcessing = atom<boolean>(false);
  $config = atom<FilterConfig>(IMAGE_FILTERS.canny_image_processor.buildDefaults());

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating filter module');
  }

  startFilter = (entityIdentifier: CanvasEntityIdentifier) => {
    this.log.trace('Initializing filter');
    const adapter = this.manager.getAdapter(entityIdentifier);
    if (!adapter) {
      this.log.warn({ entityIdentifier }, 'Unable to find entity');
      return;
    }
    if (adapter.entityIdentifier.type !== 'raster_layer' && adapter.entityIdentifier.type !== 'control_layer') {
      this.log.warn({ entityIdentifier }, 'Unsupported entity type');
      return;
    }
    this.$adapter.set(adapter);
    this.manager.tool.$tool.set('view');
  };

  previewFilter = async () => {
    const adapter = this.$adapter.get();
    if (!adapter) {
      this.log.warn('Cannot preview filter without an adapter');
      return;
    }
    const config = this.$config.get();
    this.log.trace({ config }, 'Previewing filter');
    const rect = adapter.transformer.getRelativeRect();
    const imageDTO = await adapter.renderer.rasterize({ rect });
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
      adapter.renderer.clearBuffer();

      await adapter.renderer.setBuffer(this.imageState, true);

      adapter.renderer.hideObjects();
      this.$isProcessing.set(false);
    };

    this.manager.socket.on('invocation_complete', listener);

    this.log.trace({ batch } as SerializableObject, 'Enqueuing filter batch');

    this.$isProcessing.set(true);
    this.manager.stateApi.enqueueBatch(batch);
  };

  applyFilter = () => {
    const imageState = this.imageState;
    const adapter = this.$adapter.get();
    if (!imageState) {
      this.log.warn('No image state to apply filter to');
      return;
    }
    if (!adapter) {
      this.log.warn('Cannot apply filter without an adapter');
      return;
    }
    this.log.trace('Applying filter');
    adapter.renderer.commitBuffer();
    const rect = adapter.transformer.getRelativeRect();
    this.manager.stateApi.rasterizeEntity({
      entityIdentifier: adapter.entityIdentifier,
      imageObject: imageState,
      rect: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: imageState.image.height,
        height: imageState.image.width,
      },
      replaceObjects: true,
    });
    adapter.renderer.showObjects();
    this.imageState = null;
    this.$adapter.set(null);
  };

  cancelFilter = () => {
    this.log.trace('Cancelling filter');

    const adapter = this.$adapter.get();

    if (adapter) {
      adapter.renderer.clearBuffer();
      adapter.renderer.showObjects();
      adapter.transformer.updatePosition();
      adapter.renderer.syncCache(true);
      this.$adapter.set(null);
    }
    this.imageState = null;
    this.$isProcessing.set(false);
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
              // Control images are always intermediate - do not save to gallery
              // is_intermediate: true,
              is_intermediate: false, // false for testing
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
