/**
 * This file contains IDs, names, and ID getters for konva layers and objects.
 */

// IDs for singleton Konva layers and objects
export const PREVIEW_LAYER_ID = 'preview_layer';
export const PREVIEW_TOOL_GROUP_ID = `${PREVIEW_LAYER_ID}.tool_group`;
export const PREVIEW_BRUSH_GROUP_ID = `${PREVIEW_LAYER_ID}.brush_group`;
export const PREVIEW_BRUSH_FILL_ID = `${PREVIEW_LAYER_ID}.brush_fill`;
export const PREVIEW_BRUSH_BORDER_INNER_ID = `${PREVIEW_LAYER_ID}.brush_border_inner`;
export const PREVIEW_BRUSH_BORDER_OUTER_ID = `${PREVIEW_LAYER_ID}.brush_border_outer`;
export const PREVIEW_RECT_ID = `${PREVIEW_LAYER_ID}.rect`;
export const PREVIEW_GENERATION_BBOX_GROUP = `${PREVIEW_LAYER_ID}.gen_bbox_group`;
export const PREVIEW_GENERATION_BBOX_TRANSFORMER = `${PREVIEW_LAYER_ID}.gen_bbox_transformer`;
export const PREVIEW_GENERATION_BBOX_DUMMY_RECT = `${PREVIEW_LAYER_ID}.gen_bbox_dummy_rect`;
export const PREVIEW_DOCUMENT_SIZE_GROUP = `${PREVIEW_LAYER_ID}.doc_size_group`;
export const PREVIEW_DOCUMENT_SIZE_STAGE_RECT = `${PREVIEW_LAYER_ID}.doc_size_stage_rect`;
export const PREVIEW_DOCUMENT_SIZE_DOCUMENT_RECT = `${PREVIEW_LAYER_ID}.doc_size_doc_rect`;

// Names for Konva layers and objects (comparable to CSS classes)
export const LAYER_BBOX_NAME = 'layer_bbox';
export const COMPOSITING_RECT_NAME = 'compositing_rect';
export const IMAGE_PLACEHOLDER_NAME = 'image_placeholder';

export const CA_LAYER_NAME = 'control_adapter';
export const CA_LAYER_OBJECT_GROUP_NAME = `${CA_LAYER_NAME}.object_group`;
export const CA_LAYER_IMAGE_NAME = `${CA_LAYER_NAME}.image`;

export const RG_LAYER_NAME = 'regional_guidance_layer';
export const RG_LAYER_OBJECT_GROUP_NAME = `${RG_LAYER_NAME}.object_group`;
export const RG_LAYER_BRUSH_LINE_NAME = `${RG_LAYER_NAME}.brush_line`;
export const RG_LAYER_ERASER_LINE_NAME = `${RG_LAYER_NAME}.eraser_line`;
export const RG_LAYER_RECT_SHAPE_NAME = `${RG_LAYER_NAME}.rect_shape`;

export const RASTER_LAYER_NAME = 'raster_layer';
export const RASTER_LAYER_OBJECT_GROUP_NAME = `${RASTER_LAYER_NAME}.object_group`;
export const RASTER_LAYER_BRUSH_LINE_NAME = `${RASTER_LAYER_NAME}.brush_line`;
export const RASTER_LAYER_ERASER_LINE_NAME = `${RASTER_LAYER_NAME}.eraser_line`;
export const RASTER_LAYER_RECT_SHAPE_NAME = `${RASTER_LAYER_NAME}.rect_shape`;
export const RASTER_LAYER_IMAGE_NAME = `${RASTER_LAYER_NAME}.image`;

export const INPAINT_MASK_LAYER_ID = 'inpaint_mask_layer';
export const INPAINT_MASK_LAYER_OBJECT_GROUP_NAME = `${INPAINT_MASK_LAYER_ID}.object_group`;
export const INPAINT_MASK_LAYER_BRUSH_LINE_NAME = `${INPAINT_MASK_LAYER_ID}.brush_line`;
export const INPAINT_MASK_LAYER_ERASER_LINE_NAME = `${INPAINT_MASK_LAYER_ID}.eraser_line`;
export const INPAINT_MASK_LAYER_RECT_SHAPE_NAME = `${INPAINT_MASK_LAYER_ID}.rect_shape`;

export const BACKGROUND_LAYER_ID = 'background_layer';

// Getters for non-singleton layer and object IDs
export const getRGId = (entityId: string) => `${RG_LAYER_NAME}_${entityId}`;
export const getLayerId = (entityId: string) => `${RASTER_LAYER_NAME}_${entityId}`;
export const getBrushLineId = (entityId: string, lineId: string) => `${entityId}.brush_line_${lineId}`;
export const getEraserLineId = (entityId: string, lineId: string) => `${entityId}.eraser_line_${lineId}`;
export const getRectShapeId = (entityId: string, rectId: string) => `${entityId}.rect_${rectId}`;
export const getImageObjectId = (entityId: string, imageId: string) => `${entityId}.image_${imageId}`;
export const getObjectGroupId = (entityId: string, groupId: string) => `${entityId}.objectGroup_${groupId}`;
export const getLayerBboxId = (entityId: string) => `${entityId}.bbox`;
export const getCAId = (entityId: string) => `${CA_LAYER_NAME}_${entityId}`;
export const getIPAId = (entityId: string) => `ip_adapter_${entityId}`;
