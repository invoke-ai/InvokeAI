/**
 * This file contains IDs, names, and ID getters for konva layers and objects.
 */

// IDs for singleton Konva layers and objects
export const PREVIEW_LAYER_ID = 'preview_layer';
export const PREVIEW_TOOL_GROUP_ID = 'preview_layer.tool_group';
export const PREVIEW_BRUSH_GROUP_ID = 'preview_layer.brush_group';
export const PREVIEW_BRUSH_FILL_ID = 'preview_layer.brush_fill';
export const PREVIEW_BRUSH_BORDER_INNER_ID = 'preview_layer.brush_border_inner';
export const PREVIEW_BRUSH_BORDER_OUTER_ID = 'preview_layer.brush_border_outer';
export const PREVIEW_RECT_ID = 'preview_layer.rect';
export const PREVIEW_GENERATION_BBOX_GROUP = 'preview_layer.gen_bbox_group';
export const PREVIEW_GENERATION_BBOX_TRANSFORMER = 'preview_layer.gen_bbox_transformer';
export const PREVIEW_GENERATION_BBOX_DUMMY_RECT = 'preview_layer.gen_bbox_dummy_rect';
export const PREVIEW_DOCUMENT_SIZE_GROUP = 'preview_layer.doc_size_group';
export const PREVIEW_DOCUMENT_SIZE_STAGE_RECT = 'preview_layer.doc_size_stage_rect';
export const PREVIEW_DOCUMENT_SIZE_DOCUMENT_RECT = 'preview_layer.doc_size_doc_rect';

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
export const RASTER_LAYER_IMAGE_NAME = 'raster_layer.image';

export const INPAINT_MASK_LAYER_NAME = 'inpaint_mask_layer';

export const BACKGROUND_LAYER_ID = 'background_layer';

// Getters for non-singleton layer and object IDs
export const getRGId = (entityId: string) => `${RG_LAYER_NAME}_${entityId}`;
export const getLayerId = (entityId: string) => `${RASTER_LAYER_NAME}_${entityId}`;
export const getBrushLineId = (entityId: string, lineId: string) => `${entityId}.brush_line_${lineId}`;
export const getEraserLineId = (entityId: string, lineId: string) => `${entityId}.eraser_line_${lineId}`;
export const getRectShapeId = (entityId: string, rectId: string) => `${entityId}.rect_${rectId}`;
export const getImageObjectId = (entityId: string, imageName: string) => `${entityId}.image_${imageName}`;
export const getObjectGroupId = (entityId: string, groupId: string) => `${entityId}.objectGroup_${groupId}`;
export const getLayerBboxId = (entityId: string) => `${entityId}.bbox`;
export const getCAId = (entityId: string) => `control_adapter_${entityId}`;
export const getCAImageId = (entityId: string, imageName: string) => `${entityId}.image_${imageName}`;
export const getIPAId = (entityId: string) => `ip_adapter_${entityId}`;
