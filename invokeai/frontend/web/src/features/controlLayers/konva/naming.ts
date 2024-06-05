/**
 * This file contains IDs, names, and ID getters for konva layers and objects.
 */

// IDs for singleton Konva layers and objects
export const TOOL_PREVIEW_LAYER_ID = 'tool_preview_layer';
export const TOOL_PREVIEW_BRUSH_GROUP_ID = 'tool_preview_layer.brush_group';
export const TOOL_PREVIEW_BRUSH_FILL_ID = 'tool_preview_layer.brush_fill';
export const TOOL_PREVIEW_BRUSH_BORDER_INNER_ID = 'tool_preview_layer.brush_border_inner';
export const TOOL_PREVIEW_BRUSH_BORDER_OUTER_ID = 'tool_preview_layer.brush_border_outer';
export const TOOL_PREVIEW_RECT_ID = 'tool_preview_layer.rect';
export const BACKGROUND_LAYER_ID = 'background_layer';
export const BACKGROUND_RECT_ID = 'background_layer.rect';
export const NO_LAYERS_MESSAGE_LAYER_ID = 'no_layers_message';

// Names for Konva layers and objects (comparable to CSS classes)
export const LAYER_BBOX_NAME = 'layer.bbox';
export const COMPOSITING_RECT_NAME = 'compositing-rect';

export const CA_LAYER_NAME = 'control_adapter_layer';
export const CA_LAYER_IMAGE_NAME = 'control_adapter_layer.image';

export const INITIAL_IMAGE_LAYER_ID = 'singleton_initial_image_layer';
export const INITIAL_IMAGE_LAYER_NAME = 'initial_image_layer';
export const INITIAL_IMAGE_LAYER_IMAGE_NAME = 'initial_image_layer.image';

export const RG_LAYER_NAME = 'regional_guidance_layer';
export const RG_LAYER_OBJECT_GROUP_NAME = 'regional_guidance_layer.object_group';
export const RG_LAYER_BRUSH_LINE_NAME = 'regional_guidance_layer.brush_line';
export const RG_LAYER_ERASER_LINE_NAME = 'regional_guidance_layer.eraser_line';
export const RG_LAYER_RECT_SHAPE_NAME = 'regional_guidance_layer.rect_shape';

export const RASTER_LAYER_NAME = 'raster_layer';
export const RASTER_LAYER_OBJECT_GROUP_NAME = 'raster_layer.object_group';
export const RASTER_LAYER_BRUSH_LINE_NAME = 'raster_layer.brush_line';
export const RASTER_LAYER_ERASER_LINE_NAME = 'raster_layer.eraser_line';
export const RASTER_LAYER_RECT_SHAPE_NAME = 'raster_layer.rect_shape';

// Getters for non-singleton layer and object IDs
export const getRGLayerId = (layerId: string) => `${RG_LAYER_NAME}_${layerId}`;
export const getRasterLayerId = (layerId: string) => `${RASTER_LAYER_NAME}_${layerId}`;
export const getBrushLineId = (layerId: string, lineId: string) => `${layerId}.brush_line_${lineId}`;
export const getEraserLineId = (layerId: string, lineId: string) => `${layerId}.eraser_line_${lineId}`;
export const getRectId = (layerId: string, lineId: string) => `${layerId}.rect_${lineId}`;
export const getObjectGroupId = (layerId: string, groupId: string) => `${layerId}.objectGroup_${groupId}`;
export const getLayerBboxId = (layerId: string) => `${layerId}.bbox`;
export const getCALayerId = (layerId: string) => `control_adapter_layer_${layerId}`;
export const getCALayerImageId = (layerId: string, imageName: string) => `${layerId}.image_${imageName}`;
export const getIILayerImageId = (layerId: string, imageName: string) => `${layerId}.image_${imageName}`;
export const getIPALayerId = (layerId: string) => `ip_adapter_layer_${layerId}`;
