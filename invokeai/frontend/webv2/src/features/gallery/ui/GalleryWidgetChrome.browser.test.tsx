/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryUiAdapter } from '@features/gallery/react';

import { ChakraProvider } from '@chakra-ui/react';
import { GalleryUiProvider } from '@features/gallery/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GalleryWidgetHeaderActions, GalleryWidgetLabel } from './GalleryWidgetChrome';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: { language: 'en' },
    t: (key: string, values?: Record<string, unknown>) =>
      values?.name === undefined ? key : `${key}:${String(values.name)}`,
  }),
}));

const board = {
  archived: false,
  assetCount: 0,
  id: 'dogs',
  imageCount: 3,
  kind: 'board',
  name: 'dogs',
  videoCount: 0,
};

const uncategorized = { ...board, id: 'none', kind: 'uncategorized', name: '' };
const dateBoard = { ...board, id: 'by_date:2026-07-30', kind: 'date', name: '30 July' };

vi.mock('@features/gallery/data/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  galleryBoardsOptions: () => ({
    queryFn: () => Promise.resolve([board, uncategorized, dateBoard]),
    queryKey: ['test-gallery-boards'],
    staleTime: Infinity,
  }),
}));

const updateSettings = vi.fn();

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createAdapter = (values: Record<string, unknown>) =>
  ({
    gallery: { updateSettings },
    galleryValues: values,
    projectName: 'Project',
  }) as unknown as GalleryUiAdapter;

const renderChrome = async (
  Component: typeof GalleryWidgetLabel | typeof GalleryWidgetHeaderActions,
  values: Record<string, unknown> = {},
  region: 'center' | 'right' = 'right'
) => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient!}>
          <GalleryUiProvider adapter={createAdapter(values)}>
            <Component region={region} />
          </GalleryUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
  // Let the boards query resolve so the label can name the board.
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
};

const getToggle = (): HTMLButtonElement => {
  const button = host?.querySelector<HTMLButtonElement>('button[aria-expanded]');

  if (!button) {
    throw new Error('board toggle did not render');
  }

  return button;
};

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'left:0;position:fixed;top:0;width:400px;';
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  updateSettings.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  queryClient = null;
});

describe('GalleryWidgetLabel', () => {
  it('names the widget and the board in panel regions', async () => {
    await renderChrome(GalleryWidgetLabel, { selectedBoardId: 'dogs' });

    expect(host?.textContent).toContain('widgets.labels.gallery');
    expect(host?.textContent).toContain('dogs');
  });

  it('drops the widget name in the center, where the view selector already says it', async () => {
    await renderChrome(GalleryWidgetLabel, { selectedBoardId: 'dogs' }, 'center');

    expect(host?.textContent).not.toContain('widgets.labels.gallery');
    expect(host?.textContent).toContain('dogs');
  });

  it('translates the uncategorized board rather than showing the transported name', async () => {
    await renderChrome(GalleryWidgetLabel, { selectedBoardId: 'none' });

    expect(host?.textContent).toContain('widgets.gallery.uncategorized');
  });

  it('collapses the boards when expanded', async () => {
    await renderChrome(GalleryWidgetLabel, { selectedBoardId: 'dogs' });

    expect(getToggle().getAttribute('aria-expanded')).toBe('true');

    await act(async () => {
      getToggle().click();
      await Promise.resolve();
    });

    expect(updateSettings).toHaveBeenCalledWith({ boardPanelCollapsed: true });
  });

  it('expands the boards again when collapsed', async () => {
    await renderChrome(GalleryWidgetLabel, { boardPanelCollapsed: true, selectedBoardId: 'dogs' });

    expect(getToggle().getAttribute('aria-expanded')).toBe('false');

    await act(async () => {
      getToggle().click();
      await Promise.resolve();
    });

    expect(updateSettings).toHaveBeenCalledWith({ boardPanelCollapsed: false });
  });
});

describe('GalleryWidgetHeaderActions', () => {
  it('offers upload and settings in every region, since the body renders neither', async () => {
    for (const region of ['center', 'right'] as const) {
      await renderChrome(GalleryWidgetHeaderActions, { selectedBoardId: 'dogs' }, region);

      expect(host?.querySelector('button[aria-label^="widgets.gallery.upload"]'), region).not.toBeNull();
      expect(host?.querySelector('button[aria-label="widgets.gallery.settings"]'), region).not.toBeNull();
    }
  });

  it('disables upload for a date board, which cannot receive one', async () => {
    await renderChrome(GalleryWidgetHeaderActions, { selectedBoardId: 'by_date:2026-07-30' });

    const upload = host?.querySelector<HTMLButtonElement>('button[aria-label^="widgets.gallery.upload"]');

    expect(upload?.disabled).toBe(true);
  });
});
