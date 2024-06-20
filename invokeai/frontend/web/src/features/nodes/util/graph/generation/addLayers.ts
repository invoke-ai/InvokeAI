import { getStore } from 'app/store/nanostores/store';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { $nodeManager } from 'features/controlLayers/konva/renderers/renderer';
import { blobToDataURL } from 'features/controlLayers/konva/util';
import { baseLayerImageCacheChanged } from 'features/controlLayers/store/canvasV2Slice';
import type { LayerEntity } from 'features/controlLayers/store/types';
import type Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import { getImageDTO, imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

const isValidLayer = (entity: LayerEntity) => {
  return (
    entity.isEnabled &&
    // Boolean(entity.bbox) && TODO(psyche): Re-enable this check when we have a way to calculate bbox for all layers
    entity.objects.length > 0
  );
};

/**
 * Get the blobs of all regional prompt layers. Only visible layers are returned.
 * @param layerIds The IDs of the layers to get blobs for. If not provided, all regional prompt layers are used.
 * @param preview Whether to open a new tab displaying each layer.
 * @returns A map of layer IDs to blobs.
 */

const getBaseLayer = async (layers: LayerEntity[], bbox: IRect, preview: boolean = false): Promise<Blob> => {
  const manager = $nodeManager.get();
  assert(manager, 'Node manager is null');

  const stage = manager.stage.clone();

  stage.scaleX(1);
  stage.scaleY(1);
  stage.x(0);
  stage.y(0);

  const validLayers = layers.filter(isValidLayer);

  // Konva bug (?) - when iterating over the array returned from `stage.getLayers()`, if you destroy a layer, the array
  // is mutated in-place and the next iteration will skip the next layer. To avoid this, we first collect the layers
  // to delete in a separate array and then destroy them.
  // TODO(psyche): Maybe report this?
  const toDelete: Konva.Layer[] = [];

  for (const konvaLayer of stage.getLayers()) {
    const layer = validLayers.find((l) => l.id === konvaLayer.id());
    if (!layer) {
      toDelete.push(konvaLayer);
    }
  }

  for (const konvaLayer of toDelete) {
    konvaLayer.destroy();
  }

  const blob = await new Promise<Blob>((resolve) => {
    stage.toBlob({
      callback: (blob) => {
        assert(blob, 'Blob is null');
        resolve(blob);
      },
      ...bbox,
    });
  });

  if (preview) {
    const base64 = await blobToDataURL(blob);
    openBase64ImageInTab([{ base64, caption: 'base layer' }]);
  }

  stage.destroy();

  return blob;
};

export const getBaseLayerImage = async (): Promise<ImageDTO> => {
  const { dispatch, getState } = getStore();
  const state = getState();
  if (state.canvasV2.layers.baseLayerImageCache) {
    const imageDTO = await getImageDTO(state.canvasV2.layers.baseLayerImageCache.name);
    if (imageDTO) {
      return imageDTO;
    }
  }
  const blob = await getBaseLayer(state.canvasV2.layers.entities, state.canvasV2.bbox, true);
  const file = new File([blob], 'image.png', { type: 'image/png' });
  const req = dispatch(
    imagesApi.endpoints.uploadImage.initiate({ file, image_category: 'general', is_intermediate: true })
  );
  req.reset();
  const imageDTO = await req.unwrap();
  dispatch(baseLayerImageCacheChanged(imageDTO));
  return imageDTO;
};
