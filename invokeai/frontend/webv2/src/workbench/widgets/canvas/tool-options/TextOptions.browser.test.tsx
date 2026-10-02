import type { FontRecord } from '@features/fonts';
import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { ToolFormProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const mocks = vi.hoisted(() => {
  const font: FontRecord = {
    axes: [
      { default: 400, hidden: false, label: 'Weight', maximum: 900, minimum: 100, tag: 'wght' },
      { default: 0, hidden: false, label: 'Italic', maximum: 1, minimum: 0, tag: 'ital' },
      { default: 100, hidden: true, label: 'Width', maximum: 125, minimum: 75, tag: 'wdth' },
      { default: 0, hidden: false, label: 'Flat', maximum: 0, minimum: 0, tag: 'FLAT' },
    ],
    byteSize: 1024,
    contentHash: 'a'.repeat(64),
    family: 'Variable Sans',
    filename: 'variable-sans.ttf',
    id: 'variable-sans',
    instances: [{ coordinates: { wdth: 75, wght: 400 }, name: 'Condensed' }],
    label: 'Variable Sans Regular',
    scope: 'private',
    source: 'uploaded',
    style: 'normal',
    url: '/fonts/variable-sans.ttf',
    weight: 400,
  };
  return {
    font,
    fontsInfiniteQueryOptions: vi.fn(),
    useActiveColorPair: vi.fn(() => ({ background: '#000000', foreground: '#ffffff' })),
    useActiveColorCommands: vi.fn(() => ({ setPairColor: vi.fn() })),
    useActiveProjectSelector: vi.fn(() => null),
    useColorSampler: vi.fn(() => undefined),
    usePreparedCommit: vi.fn(() => vi.fn()),
    useTextEditSession: vi.fn(() => null),
    useTextOptions: vi.fn(() => ({
      align: 'left' as const,
      color: '#ffffff',
      fontFamily: 'Variable Sans',
      fontRef: {
        contentHash: 'a'.repeat(64),
        family: 'Variable Sans',
        id: 'variable-sans',
        label: 'Variable Sans Regular',
      },
      fontSize: 32,
      fontStyle: 'normal' as const,
      fontVariations: { ital: 0, wdth: 100, wght: 650 },
      fontWeight: 650,
      lineHeight: 1.2,
    })),
  };
});

vi.mock('@features/fonts', () => ({
  fontsInfiniteQueryOptions: mocks.fontsInfiniteQueryOptions,
}));
vi.mock('@workbench/WorkbenchContext', () => ({ useActiveProjectSelector: mocks.useActiveProjectSelector }));
vi.mock('@workbench/widgets/canvas/color-system/useActiveColors', () => ({
  useActiveColorCommands: mocks.useActiveColorCommands,
  useActiveColorPair: mocks.useActiveColorPair,
}));
vi.mock('@workbench/widgets/canvas/engineStoreHooks', () => ({
  useTextEditSession: mocks.useTextEditSession,
  useTextOptions: mocks.useTextOptions,
}));
vi.mock('@workbench/widgets/canvas/useColorSampler', () => ({ useColorSampler: mocks.useColorSampler }));
vi.mock('@workbench/widgets/canvas/useStructuralCommit', () => ({ usePreparedCommit: mocks.usePreparedCommit }));

import { textForm } from './textForm';
import { canonicalizeSelectedTextSource } from './TextOptions';

const i18n = createInstance();
const queryClient = new QueryClient();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: {
    en: {
      translation: {
        'widgets.canvas.toolOptions.textFont': 'Font',
        'widgets.canvas.toolOptions.textFontCustomGroup': 'Custom fonts',
        'widgets.canvas.toolOptions.textFontDefaultPreset': 'Default',
        'widgets.canvas.toolOptions.textFontHideAdvancedAxes': 'Hide hidden axes',
        'widgets.canvas.toolOptions.textFontPreset': 'Preset',
        'widgets.canvas.toolOptions.textFontResetAxes': 'Reset axes',
        'widgets.canvas.toolOptions.textFontShowAdvancedAxes': 'Show hidden axes',
        'widgets.canvas.toolOptions.textFontStyle': 'Style',
        'widgets.canvas.toolOptions.textFontStyleNormal': 'Normal',
        'widgets.canvas.toolOptions.textFontWeight': 'Weight',
        'widgets.properties.groups.font': 'Font',
        'widgets.properties.rows.size': 'Size',
        'widgets.properties.rows.weight': 'Weight',
        'widgets.properties.target.defaults': 'Defaults',
        'common.loading': 'Loading…',
        'common.retry': 'Retry',
      },
      fonts: {
        fonts: {
          couldNotLoad: 'Could not load fonts',
          loadMore: 'Load more fonts',
          loadingMore: 'Loading more fonts…',
        },
      },
    },
  },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('custom font controls', () => {
  let container: HTMLDivElement | null = null;
  let root: Root | null = null;
  const engine = null as unknown as ToolFormProps['engine'];

  beforeEach(() => {
    mocks.fontsInfiniteQueryOptions.mockReset().mockImplementation(() => ({
      getNextPageParam: () => undefined,
      initialPageParam: 0,
      queryFn: () => Promise.resolve({ items: [mocks.font], limit: 100, offset: 0, total: 1 }),
      queryKey: ['font-catalog-test'],
    }));
  });

  it('canonicalizes built-in restyles without upgrading a v3 source', () => {
    const source: Extract<CanvasLayerSourceContract, { type: 'text' }> = {
      align: 'left',
      color: '#ffffff',
      content: 'hello',
      fontFamily: 'Inter',
      fontRef: undefined,
      fontSize: 32,
      fontStyle: 'normal',
      fontVariations: {},
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text',
    };
    const canonical = canonicalizeSelectedTextSource(source);
    expect(canonical).not.toHaveProperty('fontRef');
    expect(canonical).not.toHaveProperty('fontStyle');
    expect(canonical).not.toHaveProperty('fontVariations');

    const custom = canonicalizeSelectedTextSource({
      ...source,
      fontRef: { contentHash: 'hash', family: 'Catalog Family', id: 'font-1', label: 'Catalog Regular' },
      fontStyle: 'italic',
      fontVariations: { wght: 650 },
    });
    expect(custom.fontRef?.id).toBe('font-1');
    expect(custom.fontStyle).toBe('italic');
    expect(custom.fontVariations).toEqual({ wght: 650 });
  });

  afterEach(async () => {
    await act(() => root?.unmount());
    queryClient.clear();
    container?.remove();
    container = null;
    root = null;
  });

  it('makes variation axes authoritative and reveals hidden axes on demand', async () => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    const FontGroup = textForm.groups[0]!.body;
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <QueryClientProvider client={queryClient}>
              <FontGroup engine={engine} isSurfaceInteractionLocked={false} />
            </QueryClientProvider>
          </I18nextProvider>
        </ChakraProvider>
      );
    });

    await expect.element(page.getByRole('slider', { name: 'Weight (wght)' })).toBeVisible();
    // The face's own axes own style and weight: no second control for either.
    expect(page.getByRole('combobox', { name: 'Style' }).query()).toBeNull();
    expect(page.getByRole('combobox', { name: 'Weight' }).query()).toBeNull();
    expect(page.getByRole('slider', { name: 'Width (wdth)' }).query()).toBeNull();
    // A degenerate axis has nothing to edit and never appears.
    expect(page.getByRole('slider', { name: 'Flat (FLAT)' }).query()).toBeNull();
    await expect.element(page.getByRole('button', { name: 'Show hidden axes' })).toBeVisible();

    await act(() => page.getByRole('button', { name: 'Show hidden axes' }).click());
    await expect.element(page.getByRole('slider', { name: 'Width (wdth)' })).toBeVisible();
    await expect.element(page.getByRole('combobox', { name: 'Preset' })).toBeVisible();
  });

  it('keeps a hidden weight axis editable instead of leaving weight with no control', async () => {
    const hiddenWeight: FontRecord = {
      ...mocks.font,
      axes: mocks.font.axes.map((axis) => (axis.tag === 'wght' ? { ...axis, hidden: true } : axis)),
    };
    mocks.fontsInfiniteQueryOptions.mockReset().mockImplementation(() => ({
      getNextPageParam: () => undefined,
      initialPageParam: 0,
      queryFn: () => Promise.resolve({ items: [hiddenWeight], limit: 100, offset: 0, total: 1 }),
      queryKey: ['font-catalog-test', 'hidden-weight'],
    }));
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    const FontGroup = textForm.groups[0]!.body;
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <QueryClientProvider client={queryClient}>
              <FontGroup engine={engine} isSurfaceInteractionLocked={false} />
            </QueryClientProvider>
          </I18nextProvider>
        </ChakraProvider>
      );
    });
    await expect.element(page.getByRole('slider', { name: 'Weight (wght)' })).toBeVisible();
    expect(page.getByRole('combobox', { name: 'Weight' }).query()).toBeNull();
    expect(page.getByRole('slider', { name: 'Width (wdth)' }).query()).toBeNull();
  });

  it('reveals catalog fonts beyond the first page with an explicit load-more action', async () => {
    const secondPageFont: FontRecord = {
      ...mocks.font,
      family: 'Second Page Sans',
      id: 'second-page-sans',
      label: 'Second Page Sans Regular',
    };
    const pageParams: number[] = [];
    mocks.fontsInfiniteQueryOptions.mockReset().mockImplementation(() => ({
      getNextPageParam: (_lastPage: unknown, _allPages: unknown[], lastPageParam: number) =>
        lastPageParam === 0 ? 100 : undefined,
      initialPageParam: 0,
      queryFn: ({ pageParam }: { pageParam: number }) => {
        pageParams.push(pageParam);
        return Promise.resolve({
          items: [pageParam === 0 ? mocks.font : secondPageFont],
          limit: 100,
          offset: pageParam,
          total: 101,
        });
      },
      queryKey: ['font-catalog-test', 'paged'],
    }));

    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    const FontGroup = textForm.groups[0]!.body;
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <QueryClientProvider client={queryClient}>
              <FontGroup engine={engine} isSurfaceInteractionLocked={false} />
            </QueryClientProvider>
          </I18nextProvider>
        </ChakraProvider>
      );
    });

    await expect.element(page.getByRole('button', { name: 'Load more fonts' })).toBeVisible();
    await act(() => page.getByRole('button', { name: 'Load more fonts' }).click());
    await vi.waitFor(() => expect(pageParams).toContain(100));

    await act(() => page.getByRole('combobox', { name: 'Font' }).click());
    await expect.element(page.getByRole('option', { name: 'Second Page Sans Regular', exact: true })).toBeVisible();
  });

  it('surfaces an initial catalog failure and retries it without losing built-ins', async () => {
    let attempts = 0;
    let rejectInitial: ((reason?: unknown) => void) | undefined;
    const initialCatalog = new Promise<never>((_resolve, reject) => {
      rejectInitial = reject;
    });
    mocks.fontsInfiniteQueryOptions.mockReset().mockImplementation(() => ({
      getNextPageParam: () => undefined,
      initialPageParam: 0,
      queryFn: () => {
        attempts += 1;
        return attempts === 1
          ? initialCatalog
          : Promise.resolve({ items: [mocks.font], limit: 100, offset: 0, total: 1 });
      },
      queryKey: ['font-catalog-test', 'retry'],
      retry: false,
    }));

    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    const FontGroup = textForm.groups[0]!.body;
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <QueryClientProvider client={queryClient}>
              <FontGroup engine={engine} isSurfaceInteractionLocked={false} />
            </QueryClientProvider>
          </I18nextProvider>
        </ChakraProvider>
      );
    });

    await expect.element(page.getByText('Loading…', { exact: true })).toBeVisible();
    await act(() => rejectInitial?.(new Error('catalog unavailable')));
    await expect.element(page.getByText('Could not load fonts', { exact: true })).toBeVisible();
    await expect.element(page.getByRole('combobox', { name: 'Font' })).toBeVisible();
    await expect.element(page.getByRole('button', { name: 'Retry' })).toBeVisible();

    await act(() => page.getByRole('button', { name: 'Retry' }).click());
    await expect.element(page.getByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
    await act(() => page.getByRole('combobox', { name: 'Font' }).click());
    await expect.element(page.getByRole('option', { name: 'Variable Sans Regular', exact: true })).toBeVisible();
  });

  it('preserves loaded choices when loading the next catalog page fails', async () => {
    const secondPageFont: FontRecord = {
      ...mocks.font,
      family: 'Second Page Sans',
      id: 'second-page-sans-retry',
      label: 'Second Page Sans Retry',
    };
    let failedNextPage = false;
    const pageParams: number[] = [];
    mocks.fontsInfiniteQueryOptions.mockReset().mockImplementation(() => ({
      getNextPageParam: (_lastPage: unknown, _allPages: unknown[], lastPageParam: number) =>
        lastPageParam === 0 ? 100 : undefined,
      initialPageParam: 0,
      queryFn: ({ pageParam }: { pageParam: number }) => {
        pageParams.push(pageParam);
        if (pageParam === 100 && !failedNextPage) {
          failedNextPage = true;
          return Promise.reject(new Error('next page unavailable'));
        }
        return Promise.resolve({
          items: [pageParam === 0 ? mocks.font : secondPageFont],
          limit: 100,
          offset: pageParam,
          total: 101,
        });
      },
      queryKey: ['font-catalog-test', 'next-page-retry'],
      retry: false,
    }));

    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    const FontGroup = textForm.groups[0]!.body;
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <QueryClientProvider client={queryClient}>
              <FontGroup engine={engine} isSurfaceInteractionLocked={false} />
            </QueryClientProvider>
          </I18nextProvider>
        </ChakraProvider>
      );
    });

    await expect.element(page.getByRole('button', { name: 'Load more fonts' })).toBeVisible();
    await act(() => page.getByRole('button', { name: 'Load more fonts' }).click());
    await vi.waitFor(() => expect(pageParams).toContain(100));
    await expect.element(page.getByText('Could not load fonts', { exact: true })).toBeVisible();

    await act(() => page.getByRole('combobox', { name: 'Font' }).click());
    await expect.element(page.getByRole('option', { name: 'Variable Sans Regular', exact: true })).toBeVisible();
    await act(() => userEvent.keyboard('{Escape}'));

    await act(() => page.getByRole('button', { name: 'Retry' }).click());
    await act(() => page.getByRole('combobox', { name: 'Font' }).click());
    await expect.element(page.getByRole('option', { name: 'Second Page Sans Retry', exact: true })).toBeVisible();
  });
});
