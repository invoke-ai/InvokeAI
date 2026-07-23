import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18n from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import { closeCommandPalette, openCommandPalette, useIsCommandPaletteOpen } from './paletteStore';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let diagnostics: string[] = [];
let consoleErrorSpy: ReturnType<typeof vi.spyOn> | null = null;
let consoleWarnSpy: ReturnType<typeof vi.spyOn> | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const i18nReady = fetch('/locales/en.json')
  .then((response) => response.json())
  .then((translation) =>
    i18n.use(initReactI18next).init({
      fallbackLng: 'en',
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation } },
    })
  );

const waitFor = async <Value,>(assertion: () => Value): Promise<Value> => {
  let value: Value | undefined;

  await act(async () => {
    value = await vi.waitFor(assertion);
  });

  return value as Value;
};

const waitForDebounce = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 250);
      })
  );

const formatDiagnostic = (value: unknown): string =>
  value instanceof Error ? (value.stack ?? value.message) : String(value);
const onPageError = (event: ErrorEvent) =>
  diagnostics.push(`page error: ${formatDiagnostic(event.error ?? event.message)}`);
const onUnhandledRejection = (event: PromiseRejectionEvent) =>
  diagnostics.push(`unhandled rejection: ${formatDiagnostic(event.reason)}`);

beforeEach(() => {
  diagnostics = [];
  consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation((...args) => {
    diagnostics.push(`console.error: ${args.map(formatDiagnostic).join(' ')}`);
  });
  consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation((...args) => {
    diagnostics.push(`console.warn: ${args.map(formatDiagnostic).join(' ')}`);
  });
  window.addEventListener('error', onPageError);
  window.addEventListener('unhandledrejection', onUnhandledRejection);
});

const entry = (id: string, title: string, run = vi.fn()): PaletteEntry => ({
  group: 'Commands',
  id,
  isPersistentRecent: true,
  run,
  showInEmptyState: true,
  title,
});

const storeHostEntries = [entry('first', 'First command')];

const StorePaletteHost = () => {
  const isOpen = useIsCommandPaletteOpen();

  return (
    <>
      <button type="button" onClick={openCommandPalette}>
        Open palette
      </button>
      {isOpen ? (
        <CommandPaletteDialog entries={storeHostEntries} isOpen modifierKeyLabel="ctrl" onClose={closeCommandPalette} />
      ) : null}
    </>
  );
};

const renderPalette = async ({
  entries,
  onClose = vi.fn(),
  providers,
}: {
  entries: PaletteEntry[];
  onClose?: () => void;
  providers?: PaletteSearchProvider[];
}) => {
  await i18nReady;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <CommandPaletteDialog
              entries={entries}
              isOpen
              modifierKeyLabel="ctrl"
              providers={providers}
              onClose={onClose}
            />
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => resolve());
    });
  });

  const input = await waitFor(() => {
    const node = document.querySelector<HTMLInputElement>('[name="command-palette-query"]');
    expect(node).not.toBeNull();
    return node!;
  });

  return { input, onClose };
};

afterEach(async () => {
  await act(async () => {
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => resolve());
    });
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = null;
  root = null;
  window.localStorage.clear();
  window.removeEventListener('error', onPageError);
  window.removeEventListener('unhandledrejection', onUnhandledRejection);
  consoleErrorSpy?.mockRestore();
  consoleWarnSpy?.mockRestore();
  consoleErrorSpy = null;
  consoleWarnSpy = null;
  closeCommandPalette();
  expect(diagnostics, diagnostics.join('\n\n')).toEqual([]);
});

describe('CommandPaletteDialog interaction', () => {
  it('restores focus through the store-driven host lifecycle and preserves the first return target', async () => {
    await i18nReady;
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <QueryClientProvider client={new QueryClient()}>
              <StorePaletteHost />
            </QueryClientProvider>
          </ChakraProvider>
        </I18nextProvider>
      );
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });
    });
    const button = document.querySelector<HTMLButtonElement>('button');
    expect(button).not.toBeNull();

    await act(() => userEvent.click(button!));
    const input = await waitFor(() => {
      const node = document.querySelector<HTMLInputElement>('[name="command-palette-query"]');
      expect(node).not.toBeNull();
      return node!;
    });
    expect(document.activeElement).toBe(input);

    await act(() => openCommandPalette());
    await act(() => userEvent.keyboard('{Escape}'));
    await waitFor(() => expect(document.activeElement).toBe(button));
  });

  it('exposes an accessible focused search field and runs the stable highlighted command in bare > mode', async () => {
    const firstRun = vi.fn();
    const secondRun = vi.fn();
    const providerSearch = vi.fn(() => []);
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Entities',
      providerKey: 'entities',
      search: providerSearch,
    };
    const setting: PaletteEntry = {
      group: 'Settings',
      id: 'setting.test',
      isPersistentRecent: true,
      run: vi.fn(),
      title: 'Hidden setting',
    };
    const { input, onClose } = await renderPalette({
      entries: [entry('first', 'First command', firstRun), entry('second', 'Second command', secondRun), setting],
      providers: [provider],
    });

    expect(document.activeElement).toBe(input);
    expect(input.type).toBe('search');
    expect(input.autocomplete).toBe('off');
    expect(input.spellcheck).toBe(false);
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('Command palette');
    expect(input.getAttribute('aria-controls')).toBe('command-palette-results');

    await act(() => userEvent.fill(input, '>'));
    expect(document.body.textContent).toContain('First command');
    expect(document.body.textContent).toContain('Second command');
    expect(document.body.textContent).not.toContain('Hidden setting');

    await waitForDebounce();
    expect(providerSearch).not.toHaveBeenCalled();

    await act(() => userEvent.keyboard('{ArrowDown}{Enter}'));
    expect(firstRun).not.toHaveBeenCalled();
    expect(secondRun).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('allows Tab to reach Retry and explicitly returns focus after retrying', async () => {
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Entities',
      providerKey: 'entities',
      search: vi.fn(() => Promise.reject(new Error('offline'))),
    };
    const { input } = await renderPalette({ entries: [], providers: [provider] });

    await act(() => userEvent.fill(input, 'zz'));
    await act(() => userEvent.keyboard('{Tab}'));
    const retry = await waitFor(() => {
      const node = [...document.querySelectorAll('button')].find((button) => button.textContent === 'Retry');
      expect(node).toBeDefined();
      return node!;
    });

    await act(() => userEvent.tab());
    expect(document.activeElement).toBe(retry);

    await act(() => userEvent.click(retry));
    await waitFor(() => expect(document.activeElement).toBe(input));
  });

  it('keeps successful root sections and retries only a failed provider', async () => {
    const retryResult = entry('recovered', 'Recovered result');
    const failedSearch = vi.fn().mockRejectedValueOnce(new Error('failed')).mockResolvedValueOnce([retryResult]);
    const successfulSearch = vi.fn(() => [entry('success', 'Successful result')]);
    const providers: PaletteSearchProvider[] = [
      { contextKey: 'failed', label: 'Broken', providerKey: 'broken', search: failedSearch },
      { contextKey: 'success', label: 'Working', providerKey: 'working', search: successfulSearch },
    ];
    const { input, onClose } = await renderPalette({ entries: [], providers });

    await act(() => userEvent.fill(input, 'zz'));
    await waitForDebounce();
    const retryOption = await waitFor(() => {
      expect(document.body.textContent).toContain('Successful result');
      const node = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((option) =>
        option.textContent?.includes('Retry broken search')
      );
      expect(node).toBeDefined();
      return node!;
    });

    await act(() => userEvent.click(retryOption));
    await waitFor(() => expect(document.body.textContent).toContain('Recovered result'));
    expect(failedSearch).toHaveBeenCalledTimes(2);
    expect(successfulSearch).toHaveBeenCalledTimes(1);
    expect(onClose).not.toHaveBeenCalled();
  });

  it('marks the listbox busy while a provider is pending and announces completion', async () => {
    let resolveSearch: ((entries: PaletteEntry[]) => void) | undefined;
    const search = vi.fn(
      () =>
        new Promise<PaletteEntry[]>((resolve) => {
          resolveSearch = resolve;
        })
    );
    const { input } = await renderPalette({
      entries: [],
      providers: [{ contextKey: 'pending', label: 'Entities', providerKey: 'entities', search }],
    });

    await act(() => userEvent.fill(input, 'zz'));
    const listbox = await waitFor(() => {
      const node = document.querySelector<HTMLElement>('[role="listbox"]');
      expect(node?.getAttribute('aria-busy')).toBe('true');
      expect(document.body.textContent).toContain('Searching…');
      return node!;
    });

    await waitForDebounce();
    await waitFor(() => expect(search).toHaveBeenCalledOnce());
    await act(() => resolveSearch?.([entry('found', 'Found result')]));
    await waitFor(() => expect(listbox.getAttribute('aria-busy')).toBe('false'));
    expect(document.body.textContent).toContain('1 results.');
  });

  it('does not expose or activate provider entries from the pre-debounce query', async () => {
    const oldRun = vi.fn();
    const search = vi.fn((query: { text: string }) =>
      query.text === 'old' ? [entry('old-result', 'Old result', oldRun)] : []
    );
    const { input } = await renderPalette({
      entries: [],
      providers: [{ contextKey: 'context', label: 'Entities', providerKey: 'entities', search }],
    });

    await act(() => userEvent.fill(input, 'old'));
    await waitForDebounce();
    await waitFor(() => expect(document.body.textContent).toContain('Old result'));

    await act(() => userEvent.fill(input, 'new'));
    expect(document.body.textContent).not.toContain('Old result');
    await act(() => userEvent.keyboard('{Enter}'));
    expect(oldRun).not.toHaveBeenCalled();
  });

  it('renders no interactive descendants inside listbox options', async () => {
    const withSecondary: PaletteEntry = {
      ...entry('first', 'First command'),
      secondary: { label: 'Open Elsewhere', run: vi.fn() },
    };
    await renderPalette({ entries: [withSecondary] });

    for (const option of document.querySelectorAll('[role="option"]')) {
      expect(option.querySelector('button, a, input, select, textarea')).toBeNull();
    }
  });

  it('shows key-specific date suggestions while typing a token, without flashing an error', async () => {
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Images',
      providerKey: 'images',
      search: vi.fn(() => []),
      supportsCreatedAtRange: true,
    };
    const { input } = await renderPalette({ entries: [entry('first', 'First command')], providers: [provider] });

    await act(() => userEvent.fill(input, 'from:'));

    expect(document.body.textContent).toContain('Today');
    expect(document.body.textContent).toContain('Yesterday');
    expect(document.body.textContent).toContain('Past week');
    expect(document.body.textContent).toContain('Or type a date');
    expect(document.body.textContent).not.toContain('Invalid date');
    expect(input.getAttribute('aria-invalid')).toBeNull();

    // `to:` must not offer a span-shaped completion.
    await act(() => userEvent.fill(input, 'to:'));
    expect(document.body.textContent).not.toContain('Past week');
  });

  it('completes a suggestion in place and immediately queries providers with the structured query', async () => {
    const providerSearch = vi.fn(() => []);
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Images',
      providerKey: 'images',
      search: providerSearch,
      supportsCreatedAtRange: true,
    };
    const { input, onClose } = await renderPalette({ entries: [], providers: [provider] });

    await act(() => userEvent.fill(input, 'from:y'));
    await act(() => userEvent.keyboard('{Enter}'));

    expect(input.value).toBe('from:yesterday ');
    expect(onClose).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith(
        {
          range: { from: expect.stringMatching(/^\d{4}-\d{2}-\d{2}$/), to: undefined },
          text: '',
        },
        { signal: expect.any(AbortSignal) }
      );
    });
    expect(document.body.textContent).toContain('Search images by date');
  });

  it('fires only range-capable providers for a pure date query', async () => {
    const capableSearch = vi.fn(() => []);
    const dateLessSearch = vi.fn(() => []);
    const capable: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Images',
      providerKey: 'images',
      search: capableSearch,
      supportsCreatedAtRange: true,
    };
    const dateLess: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Workflows',
      providerKey: 'workflows',
      search: dateLessSearch,
    };
    const { input } = await renderPalette({ entries: [], providers: [capable, dateLess] });

    await act(() => userEvent.fill(input, 'from:7d'));
    await waitForDebounce();

    await waitFor(() => expect(capableSearch).toHaveBeenCalled());
    expect(dateLessSearch).not.toHaveBeenCalled();
  });

  it('renders invalid date feedback with combobox aria wiring once the token is settled', async () => {
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Images',
      providerKey: 'images',
      search: vi.fn(() => []),
      supportsCreatedAtRange: true,
    };
    const { input } = await renderPalette({ entries: [], providers: [provider] });

    // The trailing space settles the token — no longer an in-progress prefix.
    await act(() => userEvent.fill(input, 'from:lastweek '));

    const hint = document.getElementById('command-palette-date-hint');
    expect(hint?.textContent).toContain('Invalid date: “lastweek”');
    expect(hint?.getAttribute('role')).toBe('status');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toBe('command-palette-date-hint');
  });

  it('treats date tokens as literal text when no range-capable provider is present', async () => {
    const dateLessSearch = vi.fn(() => []);
    const dateLess: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Workflows',
      providerKey: 'workflows',
      search: dateLessSearch,
    };
    const { input } = await renderPalette({ entries: [], providers: [dateLess] });

    await act(() => userEvent.fill(input, 'from:'));
    expect(document.body.textContent).not.toContain('Or type a date');

    await act(() => userEvent.fill(input, 'from:7d'));
    await waitForDebounce();

    // No acknowledgment chip, and the provider receives the literal text.
    expect(document.getElementById('command-palette-date-hint')).toBeNull();
    await waitFor(() => {
      expect(dateLessSearch).toHaveBeenCalledWith(
        { range: undefined, text: 'from:7d' },
        { signal: expect.any(AbortSignal) }
      );
    });
  });

  it('shows one search-scope command per provider in the empty launcher and bare > mode', async () => {
    const providers: PaletteSearchProvider[] = [
      { contextKey: 'context', label: 'Entities', providerKey: 'entities', search: vi.fn(() => []) },
      { contextKey: 'context', label: 'Widgets', providerKey: 'widgets', search: vi.fn(() => []) },
    ];
    const { input } = await renderPalette({ entries: [entry('first', 'First command')], providers });

    expect(document.body.textContent).toContain('Search in');
    expect(document.body.textContent).toContain('Search entities…');
    expect(document.body.textContent).toContain('Search widgets…');

    await act(() => userEvent.fill(input, '>'));
    expect(document.body.textContent).toContain('Search entities…');
    expect(document.body.textContent).toContain('Search widgets…');
  });

  it('shows no search-scope section without providers (Launchpad shape)', async () => {
    const { input } = await renderPalette({ entries: [entry('first', 'First command')] });

    expect(document.body.textContent).not.toContain('Search in');

    await act(() => userEvent.fill(input, '>'));
    expect(document.body.textContent).not.toContain('Search in');
  });

  it('runs a search-scope command: clears the query, keeps the palette open, and lists results immediately', async () => {
    const providerSearch = vi.fn(() => [
      { group: 'Entities', id: 'entity-one', isPersistentRecent: false, run: vi.fn(), title: 'Entity One' },
    ]);
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Entities',
      providerKey: 'entities',
      search: providerSearch,
      supportsCreatedAtRange: true,
    };
    const { input, onClose } = await renderPalette({ entries: [], providers: [provider] });

    await act(() => userEvent.fill(input, 'search ent'));
    await act(() => userEvent.keyboard('{Enter}'));

    expect(input.value).toBe('');
    expect(onClose).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith({ range: undefined, text: '' }, { signal: expect.any(AbortSignal) });
    });
    expect(document.body.textContent).toContain('Entity One');

    // The same command works from > commands mode and exits it into the scope.
    await act(() => userEvent.keyboard('{Backspace}'));
    await act(() => userEvent.fill(input, '>search ent'));
    await act(() => userEvent.keyboard('{Enter}'));
    expect(input.value).toBe('');
    await waitFor(() => expect(document.body.textContent).toContain('Entity One'));
  });

  it('keeps the query when entering a scope through a Tab scope row', async () => {
    const providerSearch = vi.fn(() => []);
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Entities',
      providerKey: 'entities',
      search: providerSearch,
    };
    const { input } = await renderPalette({ entries: [], providers: [provider] });

    await act(() => userEvent.fill(input, 'zz'));
    // The trailing scope row is the last navigable row; ArrowUp wraps to it.
    await act(() => userEvent.keyboard('{ArrowUp}'));
    await act(() => userEvent.keyboard('{Tab}'));

    expect(input.value).toBe('zz');
    await waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith(
        { range: undefined, text: 'zz' },
        { signal: expect.any(AbortSignal) }
      );
    });
  });

  it('searches at any length inside a scope but keeps the root minimum length', async () => {
    const providerSearch = vi.fn(() => []);
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Entities',
      providerKey: 'entities',
      search: providerSearch,
    };
    const { input } = await renderPalette({ entries: [], providers: [provider] });

    await act(() => userEvent.fill(input, 'e'));
    await waitForDebounce();
    expect(providerSearch).not.toHaveBeenCalled();

    await act(() => userEvent.fill(input, ''));
    await act(() => userEvent.keyboard('{Enter}'));
    await waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith({ range: undefined, text: '' }, { signal: expect.any(AbortSignal) });
    });

    await act(() => userEvent.fill(input, 'a'));
    await waitForDebounce();
    await waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith({ range: undefined, text: 'a' }, { signal: expect.any(AbortSignal) });
    });
  });

  it('says what an empty scope is missing and falls back to no-results for a miss', async () => {
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Entities',
      providerKey: 'entities',
      search: vi.fn(() => []),
    };
    const { input } = await renderPalette({ entries: [], providers: [provider] });

    await act(() => userEvent.keyboard('{Enter}'));
    await waitFor(() => expect(document.body.textContent).toContain('No entities yet'));

    await act(() => userEvent.fill(input, 'x'));
    await waitForDebounce();
    await waitFor(() => expect(document.body.textContent).toContain('No results for “x”'));
  });

  it('keeps the error state and Retry for an empty scope', async () => {
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Entities',
      providerKey: 'entities',
      search: vi.fn(() => Promise.reject(new Error('offline'))),
    };
    await renderPalette({ entries: [], providers: [provider] });

    await act(() => userEvent.keyboard('{Enter}'));
    await waitFor(() => expect(document.body.textContent).toContain("Couldn't search entities"));
    expect([...document.querySelectorAll('button')].some((button) => button.textContent === 'Retry')).toBe(true);
  });

  it('records a used search-scope command as a persistent recent', async () => {
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Entities',
      providerKey: 'entities',
      search: vi.fn(() => []),
    };
    await renderPalette({ entries: [], providers: [provider] });

    await act(() => userEvent.keyboard('{Enter}'));
    await act(() => root?.unmount());
    host?.remove();

    await renderPalette({ entries: [], providers: [provider] });
    expect(document.body.textContent).toContain('Recent');
    // Listed once, under Recent — the launcher groups don't repeat recents.
    expect(document.body.textContent?.match(/Search entities…/g)).toHaveLength(1);
  });

  it('advertises the query syntax in the empty launcher, scaled to host capabilities', async () => {
    const capable: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Images',
      providerKey: 'images',
      search: vi.fn(() => []),
      supportsCreatedAtRange: true,
    };
    await renderPalette({ entries: [entry('first', 'First command')], providers: [capable] });
    expect(document.body.textContent).toContain('Type > for commands · from:/date: to filter by date');

    await act(() => root?.unmount());
    host?.remove();

    // Launchpad shape: no providers → no date-token hint, `>` still advertised.
    await renderPalette({ entries: [entry('first', 'First command')] });
    expect(document.body.textContent).toContain('Type > for commands');
    expect(document.body.textContent).not.toContain('filter by date');
  });

  it('closes a scope from the chip and returns to the root palette with focus', async () => {
    const provider: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Entities',
      providerKey: 'entities',
      search: vi.fn(() => []),
    };
    const { input, onClose } = await renderPalette({
      entries: [entry('first', 'First command')],
      providers: [provider],
    });

    await act(() => userEvent.fill(input, 'search ent'));
    await act(() => userEvent.keyboard('{Enter}'));
    const chip = await waitFor(() => {
      const node = [...document.querySelectorAll('button')].find((button) => button.textContent?.includes('Entities'));
      expect(node).toBeDefined();
      return node!;
    });

    await act(() => userEvent.click(chip));
    expect(onClose).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain('First command');
    expect(input.value).toBe('');
    await waitFor(() => expect(document.activeElement).toBe(input));
  });

  it('runs the highlighted row secondary action with Mod+Enter and exposes no nested button', async () => {
    const secondaryRun = vi.fn();
    const withSecondary: PaletteEntry = {
      ...entry('first', 'First command'),
      secondary: { label: 'Open Elsewhere', run: secondaryRun },
    };
    const { onClose } = await renderPalette({ entries: [withSecondary] });

    expect(document.querySelector('button[aria-label="Open Elsewhere"]')).toBeNull();
    await act(() => userEvent.keyboard('{Control>}{Enter}{/Control}'));
    expect(secondaryRun).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('hands a date-less scope the token-stripped text its scope row advertised', async () => {
    const capableSearch = vi.fn(() => []);
    const dateLessSearch = vi.fn(() => []);
    const capable: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Images',
      providerKey: 'images',
      search: capableSearch,
      supportsCreatedAtRange: true,
    };
    const dateLess: PaletteSearchProvider = {
      contextKey: 'context',
      label: 'Workflows',
      providerKey: 'workflows',
      search: dateLessSearch,
    };
    const { input } = await renderPalette({ entries: [], providers: [capable, dateLess] });

    await act(() => userEvent.fill(input, 'from:7d sunset'));
    expect(document.body.textContent).toContain('Search workflows for “sunset”');

    // The workflows scope row is the last navigable row; ArrowUp wraps to it.
    await act(() => userEvent.keyboard('{ArrowUp}'));
    await act(() => userEvent.keyboard('{Tab}'));

    expect(input.value).toBe('sunset');
    await waitFor(() => {
      expect(dateLessSearch).toHaveBeenCalledWith(
        { range: undefined, text: 'sunset' },
        { signal: expect.any(AbortSignal) }
      );
    });
  });

  it('drives stage preview from navigation and restores it on Escape and unmount', async () => {
    const preview = vi.fn();
    const clearPreview = vi.fn();
    const staged: PaletteEntry = {
      ...entry('theme', 'Theme'),
      stage: {
        clearPreview,
        options: [
          { apply: vi.fn(), id: 'a', isCurrent: false, label: 'Alpha' },
          { apply: vi.fn(), id: 'b', isCurrent: true, label: 'Beta' },
        ],
        preview,
        title: 'Theme',
      },
    };
    const { input } = await renderPalette({ entries: [staged] });

    await act(() => userEvent.keyboard('{Enter}{ArrowUp}'));
    expect(preview).toHaveBeenLastCalledWith('a');

    await act(() => userEvent.keyboard('{Escape}'));
    expect(clearPreview).toHaveBeenCalledOnce();
    expect(document.activeElement).toBe(input);

    await act(() => userEvent.keyboard('{Enter}'));
    await act(() => root?.unmount());
    root = null;
    expect(clearPreview).toHaveBeenCalledTimes(2);
  });
});
