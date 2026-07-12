import type { LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { FilterOperationSessionState } from '@workbench/widgets/layers/filterOperationSession';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { FilterPanelHeading, getFilterActionEligibility, getFilterSaveTargetEligibility } from './FilterOptions';

const state = (patch: Partial<FilterOperationSessionState> = {}): FilterOperationSessionState => ({
  draft: { settings: {}, type: 'canny_edge_detection' },
  error: null,
  initialFilter: null,
  layerId: 'layer-1',
  layerName: 'Portrait',
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
    const eligibility = getFilterActionEligibility(state({ preview }));
    expect(eligibility).toMatchObject({ canApply: true, canSave: true });
    expect(getFilterSaveTargetEligibility(eligibility)).toEqual({ control: true, raster: true });
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

  it('disables mutating actions under an external interaction lock but preserves Cancel', () => {
    const eligibility = getFilterActionEligibility(state(), true);
    expect(eligibility).toEqual({
      canApply: false,
      canCancel: true,
      canEdit: false,
      canProcess: false,
      canReset: false,
      canSave: false,
    });
    expect(getFilterSaveTargetEligibility(eligibility)).toEqual({ control: false, raster: false });
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
              model: {
                base: 'any',
                hash: 'blake3-hash',
                key: 'upscale',
                name: 'Upscaler',
                type: 'spandrel_image_to_image',
              },
            },
            type: 'spandrel_filter',
          },
        })
      )
    ).toMatchObject({ canProcess: true });
  });

  it('disables Process for stale partial Spandrel identifiers', () => {
    expect(
      getFilterActionEligibility(
        state({
          draft: {
            settings: {
              model: { base: 'any', hash: '', key: 'upscale', name: 'Upscaler', type: 'spandrel_image_to_image' },
            },
            type: 'spandrel_filter',
          },
        })
      )
    ).toMatchObject({ canProcess: false });
  });
});

describe('FilterPanelHeading', () => {
  it('identifies the filter target by its captured user-facing name and type', () => {
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <FilterPanelHeading layerName="Portrait" layerTypeLabel="Raster layer" title="Filter" />
      </ChakraProvider>
    );

    expect(markup).toContain('Filter');
    expect(markup).toContain('Raster layer');
    expect(markup).toContain('Portrait');
    expect(markup).not.toContain('layer-1');
  });
});
