import { DEBOUNCE_MS } from 'features/controlLayers/konva/constants';
import { TOOL_PREVIEW_LAYER_ID } from 'features/controlLayers/konva/naming';
import { updateBboxes } from 'features/controlLayers/konva/renderers/bbox';
import { renderCALayer } from 'features/controlLayers/konva/renderers/caLayer';
import { renderIILayer } from 'features/controlLayers/konva/renderers/iiLayer';
import { renderRasterLayer } from 'features/controlLayers/konva/renderers/rasterLayer';
import { renderRGLayer } from 'features/controlLayers/konva/renderers/rgLayer';
import { renderToolPreview } from 'features/controlLayers/konva/renderers/toolPreview';
import { mapId, selectRenderableLayers } from 'features/controlLayers/konva/util';
import type { Layer, Tool } from 'features/controlLayers/store/types';
import {
  isControlAdapterLayer,
  isInitialImageLayer,
  isRasterLayer,
  isRegionalGuidanceLayer,
  isRenderableLayer,
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
 * @param layerStates Array of all layer states
 * @param globalMaskLayerOpacity The global mask layer opacity
 * @param tool The current tool
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 * @param onLayerPosChanged Callback for when the layer's position changes
 */
const renderLayers = (
  stage: Konva.Stage,
  layerStates: Layer[],
  globalMaskLayerOpacity: number,
  tool: Tool,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
): void => {
  const layerIds = layerStates.filter(isRenderableLayer).map(mapId);
  // Remove un-rendered layers
  for (const konvaLayer of stage.find<Konva.Layer>(selectRenderableLayers)) {
    if (!layerIds.includes(konvaLayer.id())) {
      konvaLayer.destroy();
    }
  }
  // We'll need to ensure the tool preview layer is on top of the rest of the layers
  let toolLayerZIndex = 0;
  for (const layer of layerStates) {
    if (isRegionalGuidanceLayer(layer)) {
      renderRGLayer(stage, layer, globalMaskLayerOpacity, tool, onLayerPosChanged);
    }
    if (isControlAdapterLayer(layer)) {
      renderCALayer(stage, layer, getImageDTO);
    }
    if (isInitialImageLayer(layer)) {
      renderIILayer(stage, layer, getImageDTO);
    }
    if (isRasterLayer(layer)) {
      renderRasterLayer(stage, layer, tool, onLayerPosChanged);
    }
    // IP Adapter layers are not rendered
    // Increment the z-index for the tool layer
    toolLayerZIndex++;
  }
  // Arrange the tool preview layer
  stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.zIndex(toolLayerZIndex);
};

/**
 * All the renderers for the Konva stage.
 */
export const renderers = {
  renderToolPreview,
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
  renderLayers: debounce(renderLayers, ms),
  updateBboxes: debounce(updateBboxes, ms),
});

/**
 * All the renderers for the Konva stage, debounced.
 */
export const debouncedRenderers: typeof renderers = getDebouncedRenderers();
