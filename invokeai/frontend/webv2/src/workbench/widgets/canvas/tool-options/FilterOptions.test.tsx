import type { LayerExportGuard } from '@workbench/canvas-engine/engine';
import type { FilterOperationSessionState } from '@workbench/widgets/layers/filterOperationSession';
import type { ComponentProps } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it, vi } from 'vitest';

import {
  FilterOptionsBar,
  getFilterActionEligibility,
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
  initImmediate: false,
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

describe('FilterOptionsBar', () => {
  it('server-renders one row of controls in stable order without panel slots', () => {
    const engine = {
      cancelFilterOperation: vi.fn(),
      commitFilterOperation: vi.fn(),
      processFilterOperation: vi.fn(),
      resetFilterOperation: vi.fn(),
      setFilterOperationAutoProcess: vi.fn(),
      updateFilterOperation: vi.fn(),
    };
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(
          I18nextProvider,
          { i18n: testI18n },
          createElement(FilterOptionsBar, {
            engine: engine as never,
            session: state({
              draft: { settings: { high_threshold: 200, low_threshold: 100 }, type: 'canny_edge_detection' },
            }),
          })
        )
      )
    );

    expect(markup).not.toContain('data-slot="header"');
    expect(markup).not.toContain('data-slot="body"');
    expect(markup).not.toContain('data-slot="feedback"');
    expect(markup).not.toContain('data-slot="footer"');
    expect(markup).not.toContain('data-operation=');
    expect(markup).toContain('Portrait · Raster layer');
    expect(markup).toContain('role="status"');
    expect(markup).toContain('aria-label="Save As"');
    expect(markup.indexOf('>Filter<')).toBeLessThan(markup.indexOf('>Auto<'));
    expect(markup.indexOf('>Auto<')).toBeLessThan(markup.indexOf('>Process<'));
    expect(markup.indexOf('>Process<')).toBeLessThan(markup.indexOf('>Reset<'));
    expect(markup.indexOf('>Reset<')).toBeLessThan(markup.indexOf('>Apply<'));
    expect(markup.indexOf('>Apply<')).toBeLessThan(markup.indexOf('aria-label="Save As"'));
    expect(markup.indexOf('aria-label="Save As"')).toBeLessThan(markup.indexOf('>Cancel<'));
  });

  it('marks the Auto chip pressed state from the session', () => {
    const render = (autoProcess: boolean) =>
      renderToStaticMarkup(
        createElement(
          ChakraProvider,
          { value: system } as ComponentProps<typeof ChakraProvider>,
          createElement(
            I18nextProvider,
            { i18n: testI18n },
            createElement(FilterOptionsBar, { engine: {} as never, session: state({ autoProcess }) })
          )
        )
      );
    expect(render(true)).toContain('aria-pressed="true"');
    expect(render(false)).toContain('aria-pressed="false"');
  });
});
