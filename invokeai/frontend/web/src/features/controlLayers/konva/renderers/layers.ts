import { DEBOUNCE_MS } from 'features/controlLayers/konva/constants';
import { BACKGROUND_LAYER_ID, TOOL_PREVIEW_LAYER_ID } from 'features/controlLayers/konva/naming';
import { renderBackground } from 'features/controlLayers/konva/renderers/background';
import { renderBboxes, updateBboxes } from 'features/controlLayers/konva/renderers/bbox';
import { renderCALayer } from 'features/controlLayers/konva/renderers/caLayer';
import { renderIILayer } from 'features/controlLayers/konva/renderers/iiLayer';
import { renderNoLayersMessage } from 'features/controlLayers/konva/renderers/noLayersMessage';
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
 * Arranges all layers in the z-axis by updating their z-indices.
 * @param stage The konva stage
 * @param layerIds An array of redux layer ids, in their z-index order
 */
const arrangeLayers = (stage: Konva.Stage, layerIds: string[]): void => {
  let nextZIndex = 0;
  // Background is the first layer
  stage.findOne<Konva.Layer>(`#${BACKGROUND_LAYER_ID}`)?.zIndex(nextZIndex++);
  // Then arrange the redux layers in order
  for (const layerId of layerIds) {
    stage.findOne<Konva.Layer>(`#${layerId}`)?.zIndex(nextZIndex++);
  }
  // Finally, the tool preview layer is always on top
  stage.findOne<Konva.Layer>(`#${TOOL_PREVIEW_LAYER_ID}`)?.zIndex(nextZIndex++);
};

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
  }
};

/**
 * All the renderers for the Konva stage.
 */
export const renderers = {
  renderToolPreview,
  renderLayers,
  renderBboxes,
  renderBackground,
  renderNoLayersMessage,
  arrangeLayers,
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
  renderBboxes: debounce(renderBboxes, ms),
  renderBackground: debounce(renderBackground, ms),
  renderNoLayersMessage: debounce(renderNoLayersMessage, ms),
  arrangeLayers: debounce(arrangeLayers, ms),
  updateBboxes: debounce(updateBboxes, ms),
});

/**
 * All the renderers for the Konva stage, debounced.
 */
export const debouncedRenderers: typeof renderers = getDebouncedRenderers();
