import type { LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { FilterOperationSessionState } from '@workbench/widgets/layers/filterOperationSession';

import { describe, expect, it } from 'vitest';

import { getFilterActionEligibility } from './FilterOptions';

const state = (patch: Partial<FilterOperationSessionState> = {}): FilterOperationSessionState => ({
  draft: { settings: {}, type: 'canny_edge_detection' },
  error: null,
  initialFilter: null,
  layerId: 'layer-1',
  layerType: 'raster',
  preview: null,
  status: 'ready',
  ...patch,
});

describe('getFilterActionEligibility', () => {
  it('allows processing/reset/cancel before a preview exists', () => {
    expect(getFilterActionEligibility(state())).toEqual({
      canApply: false,
      canCancel: true,
      canEdit: true,
      canProcess: true,
      canReset: true,
      canSave: false,
    });
  });

  it('enables apply/save only for a ready preview', () => {
    const preview = {
      guard: {} as LayerExportGuard,
      height: 10,
      imageName: 'filtered',
      origin: { x: 0, y: 0 },
      rect: { height: 10, width: 10, x: 0, y: 0 },
      width: 10,
    } as NonNullable<FilterOperationSessionState['preview']>;
    expect(getFilterActionEligibility(state({ preview }))).toMatchObject({ canApply: true, canSave: true });
  });

  it.each(['processing', 'committing'] as const)('disables ordinary actions while %s', (status) => {
    expect(getFilterActionEligibility(state({ status }))).toEqual({
      canApply: false,
      canCancel: true,
      canEdit: false,
      canProcess: false,
      canReset: false,
      canSave: false,
    });
  });

  it('disables Process for Spandrel until a compatible model is selected', () => {
    expect(
      getFilterActionEligibility(state({ draft: { settings: { model: null }, type: 'spandrel_filter' } }))
    ).toMatchObject({ canProcess: false });
    expect(
      getFilterActionEligibility(
        state({
          draft: {
            settings: {
              model: { base: 'any', key: 'upscale', name: 'Upscaler', type: 'spandrel_image_to_image' },
            },
            type: 'spandrel_filter',
          },
        })
      )
    ).toMatchObject({ canProcess: true });
  });
});
