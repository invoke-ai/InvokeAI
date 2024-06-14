import { DEBOUNCE_MS } from 'features/controlLayers/konva/constants';
import { PREVIEW_LAYER_ID } from 'features/controlLayers/konva/naming';
import { updateBboxes } from 'features/controlLayers/konva/renderers/bbox';
import { renderCALayer } from 'features/controlLayers/konva/renderers/caLayer';
import { renderIILayer } from 'features/controlLayers/konva/renderers/iiLayer';
import { renderBboxPreview, renderToolPreview } from 'features/controlLayers/konva/renderers/previewLayer';
import { renderRasterLayer } from 'features/controlLayers/konva/renderers/rasterLayer';
import { renderRGLayer } from 'features/controlLayers/konva/renderers/rgLayer';
import { mapId, selectRenderableLayers } from 'features/controlLayers/konva/util';
import type { LayerData, Tool } from 'features/controlLayers/store/types';
import {
  isControlAdapterLayer,
  isInitialImageLayer,
  isInpaintMaskLayer,
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
  layerStates: LayerData[],
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
  let zIndex = 0;
  for (const layer of layerStates) {
    if (isRegionalGuidanceLayer(layer)) {
      renderRGLayer(stage, layer, globalMaskLayerOpacity, tool, zIndex, onLayerPosChanged);
    } else if (isControlAdapterLayer(layer)) {
      renderCALayer(stage, layer, zIndex, getImageDTO);
    } else if (isInitialImageLayer(layer)) {
      renderIILayer(stage, layer, zIndex, getImageDTO);
    } else if (isRasterLayer(layer)) {
      renderRasterLayer(stage, layer, tool, zIndex, onLayerPosChanged);
    } else if (isInpaintMaskLayer(layer)) {
      //
    }
    // IP Adapter layers are not rendered
    // Increment the z-index for the tool layer
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
