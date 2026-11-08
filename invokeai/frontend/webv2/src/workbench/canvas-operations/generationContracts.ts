import type { CompositeEntry as RasterCompositeEntry, CompositeLayerRef, Rect } from '@workbench/canvas-engine/api';

export type { CompositeLayerRef, Rect };

export const DEFAULT_MASK_DENOISE_LIMIT = 1;

export type CompositeEntryKind = 'base-raster' | 'inpaint-mask' | 'noise-mask' | 'control-layer' | 'regional-mask';

export interface CompositeMaskLayerRef extends Pick<
  CompositeLayerRef,
  'id' | 'sourceRef' | 'contentSize' | 'contentOffset' | 'transform'
> {
  attributeValue: number;
}

export interface CompositeEntry extends RasterCompositeEntry {
  kind: CompositeEntryKind;
  key: string;
  maskLayers?: CompositeMaskLayerRef[];
  layerId?: string;
}

export interface CompositePlan {
  bbox: Rect;
  entries: CompositeEntry[];
}
