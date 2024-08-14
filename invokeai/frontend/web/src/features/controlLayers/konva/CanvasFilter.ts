import type { JSONObject } from 'common/types';
import { parseify } from 'common/util/serialize';
import type { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasImageState } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS, imageDTOToImageObject } from 'features/controlLayers/store/types';
import type { Logger } from 'roarr';
import { getImageDTO } from 'services/api/endpoints/images';
import { queueApi } from 'services/api/endpoints/queue';
import type { BatchConfig } from 'services/api/types';
import type { InvocationCompleteEvent } from 'services/events/types';
import { assert } from 'tsafe';

const TYPE = 'entity_filter_preview';

export class CanvasFilter {
  readonly type = TYPE;

  id: string;
  path: string[];
  parent: CanvasLayerAdapter;
  manager: CanvasManager;
  log: Logger;

  imageState: CanvasImageState | null = null;

  constructor(parent: CanvasLayerAdapter) {
    this.id = getPrefixedId(this.type);
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.parent.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.trace('Creating filter');
  }

  previewFilter = async () => {
    const { config } = this.manager.stateApi.getFilterState();
    this.log.trace({ config }, 'Previewing filter');
    const dispatch = this.manager.stateApi._store.dispatch;

    const imageDTO = await this.parent.renderer.rasterize();
    // TODO(psyche): I can't get TS to be happy, it thinkgs `config` is `never` but it should be inferred from the generic... I'll just cast it for now
    const filterNode = IMAGE_FILTERS[config.type].buildNode(imageDTO, config as never);
    const enqueueBatchArg: BatchConfig = {
      prepend: true,
      batch: {
        graph: {
          nodes: {
            [filterNode.id]: {
              ...filterNode,
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

    // Listen for the filter processing completion event
    const listener = async (event: InvocationCompleteEvent) => {
      if (event.origin !== this.id || event.invocation_source_id !== filterNode.id) {
        return;
      }
      this.log.trace({ event: parseify(event) }, 'Handling filter processing completion');
      const { result } = event;
      assert(result.type === 'image_output', `Processor did not return an image output, got: ${result}`);
      const imageDTO = await getImageDTO(result.image.image_name);
      assert(imageDTO, "Failed to fetch processor output's image DTO");
      this.imageState = imageDTOToImageObject(imageDTO);
      this.parent.renderer.clearBuffer();
      await this.parent.renderer.setBuffer(this.imageState);
      this.parent.renderer.hideObjects([this.imageState.id]);
      this.manager.socket.off('invocation_complete', listener);
    };

    this.manager.socket.on('invocation_complete', listener);

    this.log.trace({ enqueueBatchArg: parseify(enqueueBatchArg) }, 'Enqueuing filter batch');

    dispatch(
      queueApi.endpoints.enqueueBatch.initiate(enqueueBatchArg, {
        fixedCacheKey: 'enqueueBatch',
      })
    );
  };

  applyFilter = () => {
    this.log.trace('Applying filter');
    if (!this.imageState) {
      this.log.warn('No image state to apply filter to');
      return;
    }
    this.parent.renderer.commitBuffer();
    const rect = this.parent.transformer.getRelativeRect();
    this.manager.stateApi.rasterizeEntity({
      entityIdentifier: this.parent.getEntityIdentifier(),
      imageObject: this.imageState,
      rect: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: this.imageState.image.height,
        height: this.imageState.image.width,
      },
    });
    this.parent.renderer.showObjects();
    this.manager.stateApi.$filteringEntity.set(null);
    this.imageState = null;
  };

  cancelFilter = () => {
    this.log.trace('Cancelling filter');
    this.parent.renderer.clearBuffer();
    this.parent.renderer.showObjects();
    this.manager.stateApi.$filteringEntity.set(null);
    this.imageState = null;
  };

  destroy = () => {
    this.log.trace('Destroying filter');
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
    };
  };

  getLoggingContext = (): JSONObject => {
    return { ...this.parent.getLoggingContext(), path: this.path.join('.') };
  };
}
