import { DEBOUNCE_MS } from 'features/controlLayers/konva/constants';
import { BACKGROUND_LAYER_ID, PREVIEW_LAYER_ID } from 'features/controlLayers/konva/naming';
import { updateBboxes } from 'features/controlLayers/konva/renderers/bbox';
import { renderCALayer } from 'features/controlLayers/konva/renderers/caLayer';
import { renderBboxPreview, renderToolPreview } from 'features/controlLayers/konva/renderers/previewLayer';
import { renderRasterLayer } from 'features/controlLayers/konva/renderers/rasterLayer';
import { renderRGLayer } from 'features/controlLayers/konva/renderers/rgLayer';
import { mapId, selectRenderableLayers } from 'features/controlLayers/konva/util';
import type {
  CanvasEntity,
  ControlAdapterEntity,
  LayerEntity,
  PosChangedArg,
  RegionEntity,
  Tool,
} from 'features/controlLayers/store/types';
import type Konva from 'konva';
import { debounce } from 'lodash-es';
import type { ImageDTO } from 'services/api/types';

/**
 * Logic for rendering arranging and rendering all layers.
 */

/**
 * Renders the layers on the stage.
 * @param stage The konva stage
 * @param layers Array of all layer states
 * @param rgGlobalOpacity The global mask layer opacity
 * @param tool The current tool
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 * @param onPosChanged Callback for when the layer's position changes
 */
const renderLayers = (
  stage: Konva.Stage,
  layers: LayerEntity[],
  controlAdapters: ControlAdapterEntity[],
  regions: RegionEntity[],
  rgGlobalOpacity: number,
  tool: Tool,
  selectedEntity: CanvasEntity | null,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>,
  onPosChanged?: (arg: PosChangedArg, entityType: CanvasEntity['type']) => void
): void => {
  const renderableIds = [...layers.map(mapId), ...controlAdapters.map(mapId), ...regions.map(mapId)];
  // Remove un-rendered layers
  for (const konvaLayer of stage.find<Konva.Layer>(selectRenderableLayers)) {
    if (!renderableIds.includes(konvaLayer.id())) {
      konvaLayer.destroy();
    }
  }
  // We'll need to ensure the tool preview layer is on top of the rest of the layers
  let zIndex = 1;
  for (const layer of layers) {
    renderRasterLayer(stage, layer, tool, zIndex, onPosChanged);
    zIndex++;
  }
  for (const ca of controlAdapters) {
    renderCALayer(stage, ca, zIndex, getImageDTO);
    zIndex++;
  }
  for (const rg of regions) {
    renderRGLayer(stage, rg, rgGlobalOpacity, tool, zIndex, selectedEntity, onPosChanged);
    zIndex++;
  }
  // Arrange the tool preview layer
  stage.findOne<Konva.Layer>(`#${PREVIEW_LAYER_ID}`)?.zIndex(zIndex);
};

/**
 * All the renderers for the Konva stage.
 */
export const renderers = {
  renderToolPreview,
  renderBboxPreview,
  renderLayers,
  updateBboxes,
};

/**
 * Gets the renderers with debouncing applied.
 * @param ms The debounce time in milliseconds
 * @returns The renderers with debouncing applied
 */
const getDebouncedRenderers = (ms = DEBOUNCE_MS): typeof renderers => ({
  renderToolPreview: debounce(renderToolPreview, ms),
  renderBboxPreview: debounce(renderBboxPreview, ms),
  renderLayers: debounce(renderLayers, ms),
  updateBboxes: debounce(updateBboxes, ms),
});

/**
 * All the renderers for the Konva stage, debounced.
 */
export const debouncedRenderers: typeof renderers = getDebouncedRenderers();

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
