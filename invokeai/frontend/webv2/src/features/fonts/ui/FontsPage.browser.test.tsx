import { ChakraProvider } from '@chakra-ui/react';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const dependencies = vi.hoisted(() => ({
  canManageSharedFonts: false,
  deleteFont: vi.fn(),
  ensure: vi.fn(),
  fontsQueryOptions: vi.fn(),
  rescanFonts: vi.fn(),
  retain: vi.fn(() => () => undefined),
  uploadFont: vi.fn(),
}));

vi.mock('@features/fonts', () => ({
  deleteFont: dependencies.deleteFont,
  fontKeys: { all: ['fonts'] },
  fontsQueryOptions: dependencies.fontsQueryOptions,
  getFontRuntimeKey: (reference: { id: string }) => reference.id,
  rescanFonts: dependencies.rescanFonts,
  uploadFont: dependencies.uploadFont,
}));
vi.mock('@features/fonts/react', () => ({
  useFontRuntime: () => ({
    ensure: dependencies.ensure,
    retain: dependencies.retain,
    resolveFamily: () => 'Example Sans',
    subscribe: () => vi.fn(),
    getSnapshot: () => ({ generation: 0, states: new Map() }),
  }),
  useFontRuntimeSnapshot: () => ({ generation: 0, states: new Map() }),
}));
vi.mock('@features/identity', () => ({
  useCapabilities: () => ({ canManageSharedFonts: dependencies.canManageSharedFonts }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

const font = {
  axes: [],
  byteSize: 1024,
  contentHash: 'a'.repeat(64),
  family: 'Example Sans',
  filename: 'ExampleSans.ttf',
  id: 'font-a',
  instances: [],
  label: 'Example Sans Regular',
  scope: 'private',
  source: 'uploaded',
  style: 'normal',
  url: '/api/v1/fonts/font-a/file',
  weight: 400,
};

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

const pagerButton = (text: string): HTMLButtonElement | null =>
  [...host.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === text) ?? null;
let queryClient: QueryClient;

/** 960px fits the library and the detail side by side (both need 800px); 640px shows one at a time. */
const renderPage = async (width = 960): Promise<void> => {
  applyThemeToRoot(DEFAULT_THEME_ID);
  host = document.createElement('div');
  host.style.height = '720px';
  host.style.width = `${String(width)}px`;
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { FontsPage } = await import('./FontsPage');

  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <FontsPage />
        </QueryClientProvider>
      </ChakraProvider>
    );
  });
};

describe('FontsPage', () => {
  beforeEach(() => {
    dependencies.canManageSharedFonts = false;
    dependencies.deleteFont.mockReset();
    dependencies.ensure.mockReset().mockResolvedValue('Example Sans');
    dependencies.retain.mockClear();
    dependencies.fontsQueryOptions.mockReset().mockImplementation((params: unknown) => ({
      queryFn: () => Promise.resolve({ items: [font], limit: 100, offset: 0, total: 1 }),
      queryKey: ['fonts', params],
    }));
    dependencies.rescanFonts.mockReset().mockResolvedValue({ indexed: 1, revision: 1 });
    dependencies.uploadFont.mockReset();
  });

  afterEach(async () => {
    await act(() => root?.unmount());
    host?.remove();
  });

  it('shows an available font and keeps shared administration hidden for a regular account', async () => {
    await renderPage();

    await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
    expect(host.textContent).toContain('fonts.scope.private');
    expect(host.textContent).not.toContain('fonts.rescan');
    expect(host.textContent).not.toContain('fonts.uploadShared');
    expect(dependencies.fontsQueryOptions).toHaveBeenCalledWith({ limit: 100, scope: 'all', search: '' });
  });

  it('opens font details from the keyboard while showing family samples in the list', async () => {
    await renderPage();
    await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
    await vi.waitFor(() => expect(dependencies.ensure).toHaveBeenCalledOnce());
    const row = host.querySelector<HTMLButtonElement>('[role="listitem"] [data-list-primary]:not([aria-current])');
    expect(row).not.toBeNull();
    expect(row!.textContent).toContain('Aa');
    expect(dependencies.retain).not.toHaveBeenCalled();
    await act(() => {
      row!.focus();
    });
    expect(document.activeElement).toBe(row);
    await userEvent.keyboard('{Enter}');
    await vi.waitFor(() => expect(dependencies.ensure).toHaveBeenCalledTimes(2));
    expect(row!.getAttribute('aria-current')).toBe('true');
    expect(dependencies.retain).toHaveBeenCalledOnce();
    expect(host.querySelector('h3')?.textContent).toBe('Example Sans Regular');
    expect(host.textContent).toContain('fonts.previewText');
    expect(host.querySelector('button[aria-label="fonts.deleteNamed"]')).not.toBeNull();
  });

  it('exposes shared upload and directory rescan only for an administrator', async () => {
    dependencies.canManageSharedFonts = true;
    await renderPage();

    await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
    expect(host.textContent).toContain('fonts.rescan');
    const addTab = host.querySelector<HTMLButtonElement>('[role="tab"][data-value="add"]');
    expect(addTab).not.toBeNull();
    await act(() => addTab!.click());
    const sharedScopeInput = host.querySelector<HTMLInputElement>(
      '[aria-label="fonts.uploadScopeLabel"] input[value="shared"]'
    );
    expect(sharedScopeInput).not.toBeNull();
    await act(() => sharedScopeInput!.click());
    expect(host.textContent).toContain('fonts.uploadShared');

    const rescanButton = [...host.querySelectorAll('button')].find((button) =>
      button.textContent?.includes('fonts.rescan')
    );
    expect(rescanButton).toBeDefined();
    await act(async () => {
      rescanButton!.click();
      await vi.waitFor(() => expect(dependencies.rescanFonts).toHaveBeenCalledOnce());
    });
  });

  it('lets the managed catalog browse past the first page', async () => {
    const secondPageFont = { ...font, id: 'font-b', label: 'Second Page Font' };
    const requestedOffsets: number[] = [];
    dependencies.fontsQueryOptions.mockReset().mockImplementation((params: { offset?: number }) => ({
      queryFn: () => {
        const offset = params.offset ?? 0;
        requestedOffsets.push(offset);
        return Promise.resolve({
          items: [offset === 0 ? font : secondPageFont],
          limit: 100,
          offset,
          total: 101,
        });
      },
      queryKey: ['fonts', params],
    }));

    await renderPage();
    await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
    expect(requestedOffsets).toEqual([0]);

    const nextPageButton = pagerButton('common.nextPage');
    expect(nextPageButton).not.toBeNull();
    await act(() => nextPageButton!.click());
    await vi.waitFor(() => expect(host.textContent).toContain('Second Page Font'));
    expect(requestedOffsets).toEqual([0, 100]);
  });

  it('keeps navigation available when a later page becomes empty', async () => {
    const requestedOffsets: number[] = [];
    dependencies.fontsQueryOptions.mockReset().mockImplementation((params: { offset?: number }) => ({
      queryFn: () => {
        const offset = params.offset ?? 0;
        requestedOffsets.push(offset);
        return Promise.resolve({
          items: offset === 0 ? [font] : [],
          limit: 100,
          offset,
          total: offset === 0 ? 101 : 100,
        });
      },
      queryKey: ['fonts', params],
    }));

    await renderPage();
    await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
    const nextPageButton = pagerButton('common.nextPage');
    expect(nextPageButton).not.toBeNull();

    await act(() => nextPageButton!.click());
    await vi.waitFor(() => expect(host.textContent).toContain('fonts.emptyTitle'));

    const previousPageButton = pagerButton('common.previousPage');
    expect(previousPageButton).not.toBeNull();
    expect(previousPageButton!.getAttribute('aria-disabled')).toBe('false');
    expect(requestedOffsets).toEqual([0, 100]);
  });

  it('resets a scrolled library when switching to cached filter results', async () => {
    const data = {
      items: Array.from({ length: 100 }, (_, index) => ({ ...font, id: `font-${index}`, label: `Font ${index}` })),
      limit: 100,
      offset: 0,
      total: 100,
    };
    dependencies.fontsQueryOptions.mockImplementation((params: unknown) => ({
      queryFn: () => Promise.resolve(data),
      queryKey: ['fonts', params],
    }));
    await renderPage();
    await vi.waitFor(() => expect(host.querySelector('[data-index="0"]')).not.toBeNull());
    queryClient.setQueryData(['fonts', { limit: 100, scope: 'private', search: '' }], data);
    const viewport = () => host.querySelector<HTMLElement>('[data-list-viewport]')!;
    await act(() => {
      viewport().scrollTop = 800;
      viewport().dispatchEvent(new Event('scroll'));
    });
    await vi.waitFor(() => expect(viewport().scrollTop).toBe(800));
    await page.getByRole('button', { name: 'fonts.filterMenu', exact: true }).click();
    await page.getByRole('menuitemradio', { name: 'fonts.filters.my', exact: true }).click();
    // The filtered list is a fresh viewport at the top, with its first row directly under the pinned chrome.
    await vi.waitFor(() => {
      expect(viewport().scrollTop).toBe(0);
      const firstRow = host.querySelector<HTMLElement>('[data-index="0"]');
      expect(firstRow).not.toBeNull();
      expect(firstRow!.getBoundingClientRect().top).toBeCloseTo(viewport().getBoundingClientRect().top, 0);
    });
  });

  it('keeps the library rows directly below search after the page is hidden and shown', async () => {
    dependencies.fontsQueryOptions.mockImplementation((params: unknown) => ({
      queryFn: () =>
        Promise.resolve({
          items: Array.from({ length: 6 }, (_, index) => ({ ...font, id: `font-${index}`, label: `Font ${index}` })),
          limit: 100,
          offset: 0,
          total: 6,
        }),
      queryKey: ['fonts', params],
    }));
    await renderPage();
    const firstRow = () => host.querySelector<HTMLElement>('[data-index="0"] [data-list-primary]');
    await vi.waitFor(() => expect(firstRow()).not.toBeNull());
    const search = host.querySelector<HTMLInputElement>('input[aria-label="fonts.searchLabel"]')!;
    for (let index = 0; index < 3; index += 1) {
      await act(async () => {
        host.style.display = 'none';
        await new Promise<void>((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
        });
      });
      await act(async () => {
        host.style.display = 'block';
        await new Promise<void>((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
        });
      });
      await vi.waitFor(() => {
        expect(firstRow()).not.toBeNull();
        const gap = firstRow()!.getBoundingClientRect().top - search.getBoundingClientRect().bottom;
        expect(gap).toBeGreaterThanOrEqual(0);
        expect(gap).toBeLessThan(24);
      });
    }
  });

  it('serializes uploads across separately selected batches', async () => {
    let activeUploads = 0;
    let peakUploads = 0;
    const releaseUpload: Array<() => void> = [];
    dependencies.uploadFont.mockImplementation(
      () =>
        new Promise((resolve) => {
          activeUploads += 1;
          peakUploads = Math.max(peakUploads, activeUploads);
          releaseUpload.push(() => {
            activeUploads -= 1;
            resolve({ created: true });
          });
        })
    );

    await renderPage();
    await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
    const input = host.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();
    const selectFiles = (file: File): void => {
      Object.defineProperty(input!, 'files', { configurable: true, value: [file] });
      input!.dispatchEvent(new Event('change', { bubbles: true }));
    };

    await act(() => {
      selectFiles(new File(['font-a'], 'font-a.ttf', { type: 'font/ttf' }));
      selectFiles(new File(['font-b'], 'font-b.ttf', { type: 'font/ttf' }));
    });
    await vi.waitFor(() => expect(dependencies.uploadFont).toHaveBeenCalledTimes(1));
    expect(peakUploads).toBe(1);

    await act(() => releaseUpload.shift()!());
    await vi.waitFor(() => expect(dependencies.uploadFont).toHaveBeenCalledTimes(2));
    expect(peakUploads).toBe(1);

    await act(() => releaseUpload.shift()!());
    await vi.waitFor(() => expect(activeUploads).toBe(0));
  });

  it('marks queued uploads stale when the account changes before they start', async () => {
    accountLifecycle.activate('font-upload-owner-a');
    let releaseUpload: (() => void) | undefined;
    dependencies.uploadFont.mockImplementation(
      () =>
        new Promise((resolve) => {
          releaseUpload = () => resolve({ created: true });
        })
    );

    try {
      await renderPage();
      await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
      const input = host.querySelector<HTMLInputElement>('input[type="file"]');
      expect(input).not.toBeNull();
      const selectFiles = (file: File): void => {
        Object.defineProperty(input!, 'files', { configurable: true, value: [file] });
        input!.dispatchEvent(new Event('change', { bubbles: true }));
      };

      await act(() => {
        selectFiles(new File(['font-a'], 'font-a.ttf', { type: 'font/ttf' }));
        selectFiles(new File(['font-b'], 'font-b.ttf', { type: 'font/ttf' }));
      });
      await vi.waitFor(() => expect(dependencies.uploadFont).toHaveBeenCalledOnce());

      expect(releaseUpload).toBeDefined();
      accountLifecycle.activate('font-upload-owner-b');
      await act(() => releaseUpload!());
      await vi.waitFor(() => expect(host.textContent).toContain('fonts.uploadFailed'));
      expect(dependencies.uploadFont).toHaveBeenCalledOnce();
    } finally {
      accountLifecycle.invalidate();
    }
  });
  it('opens a font in a single pane, deletes it, and lands on the list', async () => {
    dependencies.deleteFont.mockResolvedValue(undefined);
    await renderPage(640);
    await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
    const back = page.getByRole('button', { name: 'fonts.backToList' });
    const library = page.getByRole('list', { name: 'fonts.library' });
    await expect.element(page.getByRole('tablist')).not.toBeInTheDocument();
    expect(await auditAccessibility(host)).toEqual([]);

    await userEvent.click(host.querySelector<HTMLElement>('[role="listitem"] [data-list-primary]')!);

    await expect.element(back).toHaveFocus();
    await expect.element(library).not.toBeInTheDocument();
    await expect.element(page.getByRole('heading', { name: 'Example Sans Regular' })).toBeVisible();
    expect(await auditAccessibility(host)).toEqual([]);
    await page.getByRole('button', { name: 'fonts.deleteNamed' }).click();
    await page.getByRole('alertdialog').getByRole('button', { name: 'fonts.deleteConfirm' }).click();

    await expect.element(page.getByRole('alertdialog')).not.toBeInTheDocument();
    await expect.element(library).toBeVisible();
    await vi.waitFor(() => {
      const focused = document.activeElement as HTMLElement;
      expect(focused.matches('[data-list-primary]')).toBe(true);
      expect(focused.checkVisibility({ visibilityProperty: true })).toBe(true);
    });
    expect(dependencies.deleteFont).toHaveBeenCalledWith('font-a');
  });

  it('opens an empty library on Add Fonts, and its empty state leads back there', async () => {
    dependencies.fontsQueryOptions.mockImplementation((params: unknown) => ({
      queryFn: () => Promise.resolve({ items: [], limit: 100, offset: 0, total: 0 }),
      queryKey: ['fonts', params],
    }));
    await renderPage(640);
    const upload = page.getByRole('button', { name: 'fonts.upload', exact: true });

    await expect.element(upload).toBeVisible();
    await page.getByRole('button', { name: 'fonts.backToList' }).click();
    await expect.element(upload).not.toBeInTheDocument();
    await page.getByRole('list', { name: 'fonts.library' }).query();
    await page.getByRole('button', { name: 'fonts.addFonts', exact: true }).first().click();

    await expect.element(upload).toBeVisible();
    await expect.element(page.getByRole('button', { name: 'fonts.backToList' })).toHaveFocus();
  });
  it.each([
    ['single-pane', 640],
    ['side by side', 960],
  ])('keeps Add Fonts, and focus, when the first upload lands (%s)', async (_mode, width) => {
    let items: (typeof font)[] = [];
    dependencies.fontsQueryOptions.mockImplementation((params: unknown) => ({
      queryFn: () => Promise.resolve({ items, limit: 100, offset: 0, total: items.length }),
      queryKey: ['fonts', params],
    }));
    dependencies.uploadFont.mockImplementation(() => {
      items = [font];
      return Promise.resolve({ created: true });
    });
    await renderPage(width);
    const addTab = page.getByRole('tab', { name: 'fonts.addFonts' });
    const upload = page.getByRole('button', { name: 'fonts.upload', exact: true });
    await expect.element(addTab).toHaveAttribute('aria-selected', 'true');
    await act(() => upload.element().focus());

    const input = host.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', {
      configurable: true,
      value: [new File(['font'], 'font.ttf', { type: 'font/ttf' })],
    });
    await act(() => input.dispatchEvent(new Event('change', { bubbles: true })));

    await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
    await expect.element(addTab).toHaveAttribute('aria-selected', 'true');
    await expect.element(upload).toHaveFocus();
  });

  it('stays on the list, now empty, after deleting the last font in a single pane', async () => {
    let items = [font];
    dependencies.fontsQueryOptions.mockImplementation((params: unknown) => ({
      queryFn: () => Promise.resolve({ items, limit: 100, offset: 0, total: items.length }),
      queryKey: ['fonts', params],
    }));
    dependencies.deleteFont.mockImplementation(() => {
      items = [];
      return Promise.resolve();
    });
    await renderPage(640);
    await vi.waitFor(() => expect(host.textContent).toContain('Example Sans Regular'));
    await userEvent.click(host.querySelector<HTMLElement>('[role="listitem"] [data-list-primary]')!);

    await page.getByRole('button', { name: 'fonts.deleteNamed' }).click();
    await page.getByRole('alertdialog').getByRole('button', { name: 'fonts.deleteConfirm' }).click();

    await expect.element(page.getByText('fonts.emptyTitle', { exact: true })).toBeVisible();
    await expect.element(page.getByRole('tab', { name: 'fonts.addFonts' })).not.toBeInTheDocument();
  });
});
