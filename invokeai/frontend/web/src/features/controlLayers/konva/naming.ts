/**
 * This file contains IDs, names, and ID getters for konva layers and objects.
 */

// Getters for non-singleton layer and object IDs
export const getRGId = (entityId: string) => `region_${entityId}`;
export const getLayerId = (entityId: string) => `layer_${entityId}`;
export const getBrushLineId = (entityId: string, lineId: string, isBuffer?: boolean) =>
  `${isBuffer ? 'buffer_' : ''}brush_line_${lineId}`;
export const getEraserLineId = (entityId: string, lineId: string, isBuffer?: boolean) =>
  `${isBuffer ? 'buffer_' : ''}eraser_line_${lineId}`;
export const getRectShapeId = (entityId: string, rectId: string, isBuffer?: boolean) =>
  `${isBuffer ? 'buffer_' : ''}rect_${rectId}`;
export const getImageObjectId = (entityId: string, imageId: string) => `image_${imageId}`;
export const getObjectGroupId = (entityId: string, groupId: string) => `objectGroup_${groupId}`;
export const getLayerBboxId = (entityId: string) => `${entityId}.bbox`;
export const getCAId = (entityId: string) => `control_adapter_${entityId}`;
export const getIPAId = (entityId: string) => `ip_adapter_${entityId}`;
