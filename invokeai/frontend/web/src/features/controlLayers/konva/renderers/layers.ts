import { BACKGROUND_LAYER_ID, PREVIEW_LAYER_ID } from 'features/controlLayers/konva/naming';
import type { ControlAdapterEntity, LayerEntity, RegionEntity } from 'features/controlLayers/store/types';
import type Konva from 'konva';

export const arrangeEntities = (
  stage: Konva.Stage,
  layers: LayerEntity[],
  controlAdapters: ControlAdapterEntity[],
  regions: RegionEntity[]
): void => {
  let zIndex = 0;
  stage.findOne<Konva.Layer>(`#${BACKGROUND_LAYER_ID}`)?.zIndex(++zIndex);
  for (const layer of layers) {
    stage.findOne<Konva.Layer>(`#${layer.id}`)?.zIndex(++zIndex);
  }
  for (const ca of controlAdapters) {
    stage.findOne<Konva.Layer>(`#${ca.id}`)?.zIndex(++zIndex);
  }
  for (const rg of regions) {
    stage.findOne<Konva.Layer>(`#${rg.id}`)?.zIndex(++zIndex);
  }
  stage.findOne<Konva.Layer>(`#${PREVIEW_LAYER_ID}`)?.zIndex(++zIndex);
};
