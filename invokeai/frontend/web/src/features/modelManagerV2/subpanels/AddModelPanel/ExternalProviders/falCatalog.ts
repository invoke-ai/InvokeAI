import type { FalCatalogModel } from 'services/api/types';

const NATIVE_CANVAS_KINDS = new Set<FalCatalogModel['kind']>(['text-to-image', 'image-to-image', 'inpaint', 'upscale']);

export const isFalNativeCanvasModel = (model: FalCatalogModel): boolean => NATIVE_CANVAS_KINDS.has(model.kind);
