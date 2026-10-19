import type { LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { FilterOperationSessionState, FilterSessionErrorCode } from '@workbench/canvas-operations/api';
import type { ComponentProps } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { CONTROL_FILTERS, FILTER_CATEGORY_ORDER } from '@workbench/canvas-operations/api';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createInstance } from 'i18next';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it, vi } from 'vitest';

import {
  filterOperationForm,
  getFilterActionEligibility,
  getFilterErrorTranslationKey,
  getFilterSaveTargetEligibility,
  getFilterStatusTranslationKey,
} from './FilterOptions';

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const enCatalog = Object.values(englishCatalogModules)[0] as Record<string, unknown>;
const testI18n = createInstance();
await testI18n.init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: enCatalog } },
});

const state = (patch: Partial<FilterOperationSessionState> = {}): FilterOperationSessionState => ({
  autoProcess: true,
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

describe('getFilterStatusTranslationKey', () => {
  it('maps each session status to its message key', () => {
    expect(getFilterStatusTranslationKey('processing')).toBe('widgets.layers.rasterFilter.running');
    expect(getFilterStatusTranslationKey('committing')).toBe('widgets.layers.rasterFilter.statusCommitting');
    expect(getFilterStatusTranslationKey('error')).toBe('widgets.layers.rasterFilter.statusError');
    expect(getFilterStatusTranslationKey('ready')).toBe('widgets.layers.selectObject.statusReady');
  });
});

describe('getFilterErrorTranslationKey', () => {
  const codes: Record<FilterSessionErrorCode, true> = {
    aborted: true,
    'apply-failed': true,
    busy: true,
    disabled: true,
    empty: true,
    locked: true,
    missing: true,
    'not-ready': true,
    'over-budget': true,
    'process-failed': true,
    'source-over-budget': true,
    stale: true,
    unsupported: true,
  };

  it.each(Object.keys(codes) as FilterSessionErrorCode[])('has distinct English copy for %s', (code) => {
    const copy = testI18n.t(getFilterErrorTranslationKey(code));
    expect(copy).not.toBe(getFilterErrorTranslationKey(code));
    expect(copy).not.toMatch(/{{/);
  });

  it('gives every code its own message', () => {
    const copies = (Object.keys(codes) as FilterSessionErrorCode[]).map((code) =>
      testI18n.t(getFilterErrorTranslationKey(code))
    );
    expect(new Set(copies).size).toBe(copies.length);
  });
});

const renderRegions = (session: FilterOperationSessionState) => {
  const operations = {
    cancelFilterOperation: vi.fn(),
    commitFilterOperation: vi.fn(),
    getFilterSessionState: () => session,
    processFilterOperation: vi.fn(),
    resetFilterOperation: vi.fn(),
    setFilterOperationAutoProcess: vi.fn(),
    subscribeFilterSession: () => () => undefined,
    updateFilterOperation: vi.fn(),
  };
  const engine = {};
  attachCanvasOperations(engine, operations as never);
  const render = (element: React.ReactElement) =>
    renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(I18nextProvider, { i18n: testI18n }, element)
      )
    );
  const [choose, params] = filterOperationForm.groups;
  return {
    choose: render(createElement(choose!.body, { engine: engine as never, isSurfaceInteractionLocked: false })),
    footer: render(
      createElement(filterOperationForm.footer, { engine: engine as never, isExternalInteractionLocked: false })
    ),
    operations,
    params: render(createElement(params!.body, { engine: engine as never, isSurfaceInteractionLocked: false })),
  };
};

describe('filter operation form', () => {
  it('puts the choice and auto switch in one group, parameters in the next, and every verb in the footer', () => {
    const { choose, footer, params } = renderRegions(
      state({ draft: { settings: { high_threshold: 200, low_threshold: 100 }, type: 'canny_edge_detection' } })
    );

    // The select names itself through a real (visually hidden) label part.
    expect(choose).toContain('>Filter</label>');
    expect(choose).toContain('>Process automatically<');
    expect(choose).not.toContain('>Process<');

    expect(params).not.toContain('>Filter</label>');
    expect(params).toContain('Low threshold');

    expect(footer).toContain('Portrait · Raster layer');
    expect(footer).toContain('role="status"');
    // Auto-process on: no Process verb; Reset then Apply then Cancel.
    expect(footer).not.toContain('>Process<');
    expect(footer.indexOf('>Reset<')).toBeLessThan(footer.indexOf('>Apply<'));
    expect(footer.indexOf('>Apply<')).toBeLessThan(footer.indexOf('>Cancel<'));
  });

  it('shows Process only with auto-process off, disabled while processing, and keeps Cancel live', () => {
    const manual = renderRegions(state({ autoProcess: false })).footer;
    expect(manual).toContain('>Process<');

    const busy = renderRegions(state({ autoProcess: false, status: 'processing' })).footer;
    const processIdx = busy.indexOf('>Process<');
    const processButtonTag = busy.slice(busy.lastIndexOf('<button', processIdx), processIdx);
    expect(processButtonTag).toContain('disabled=""');
    expect(processButtonTag).toContain('data-loading=""');
    const cancelIdx = busy.indexOf('>Cancel<');
    expect(busy.slice(busy.lastIndexOf('<button', cancelIdx), cancelIdx)).not.toContain('disabled=""');
  });

  it('reflects the auto-process switch state from the session', () => {
    const switchState = (markup: string): string | null =>
      /data-scope="switch"[^>]*data-part="root"[^>]*data-state="([a-z]+)"/.exec(markup)?.[1] ?? null;
    expect(switchState(renderRegions(state({ autoProcess: true })).choose)).toBe('checked');
    expect(switchState(renderRegions(state({ autoProcess: false })).choose)).toBe('unchecked');
  });

  it('assigns every filter a picker category covered by the display order', () => {
    // The grouped select renders in a portal, so the taxonomy is asserted here.
    for (const filter of CONTROL_FILTERS) {
      expect(FILTER_CATEGORY_ORDER, filter.type).toContain(filter.category);
    }
    expect(new Set(CONTROL_FILTERS.map((filter) => filter.category)).size).toBe(FILTER_CATEGORY_ORDER.length);
  });
});
