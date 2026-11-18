import { ChakraProvider } from '@chakra-ui/react';
import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { IS_MAC_OS } from '@workbench/hotkeys/keys';
import i18n from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

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
  const dialog = useExitPresence(isOpen);

  return (
    <>
      <button type="button" onClick={openCommandPalette}>
        Open palette
      </button>
      {dialog.isMounted ? (
        <CommandPaletteDialog
          key={dialog.generation}
          entries={storeHostEntries}
          isOpen={isOpen}
          modifierKeyLabel="ctrl"
          onClose={closeCommandPalette}
          onExitComplete={dialog.release}
        />
      ) : null}
    </>
  );
};

const renderStorePaletteHost = async () => {
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

  return button!;
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
    const button = await renderStorePaletteHost();

    await act(() => userEvent.click(button));
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
    await expect
      .element(page.getByRole('listbox', { name: 'Command palette results' }))
      .toHaveAttribute('id', 'command-palette-results');

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
    await waitForDebounce();
    await act(() => userEvent.keyboard('{ArrowUp}'));
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

    await act(() => userEvent.keyboard('{Backspace}'));
    await act(() => userEvent.fill(input, '>search ent'));
    await act(() => userEvent.keyboard('{Enter}'));
    expect(input.value).toBe('');
    await waitFor(() => expect(document.body.textContent).toContain('Entity One'));
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
    expect(document.body.textContent?.match(/Search entities…/g)).toHaveLength(1);
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

  it('moves the highlight with wrapping arrows and clamped pages, leaving Home/End to the caret', async () => {
    const entries = Array.from({ length: 30 }, (_, index) =>
      entry(`command-${index}`, `Command ${String(index).padStart(2, '0')}`)
    );
    const { input } = await renderPalette({ entries });
    const highlighted = () => document.getElementById(input.getAttribute('aria-activedescendant') ?? '');
    const expectHighlighted = (title: string) =>
      waitFor(() => {
        expect(highlighted()?.textContent).toBe(title);
        expect(highlighted()?.getAttribute('aria-selected')).toBe('true');
      });

    await expectHighlighted('Command 00');
    // Wrapping lands on a row outside the rendered window; the list scrolls it in.
    await act(() => userEvent.keyboard('{ArrowUp}'));
    await expectHighlighted('Command 29');
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await expectHighlighted('Command 00');
    await act(() => userEvent.keyboard('{PageDown}'));
    await expectHighlighted('Command 08');
    await act(() => userEvent.keyboard('{PageUp}{PageUp}'));
    await expectHighlighted('Command 00');
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await expectHighlighted('Command 01');
    expect(document.querySelectorAll('[role="option"][aria-selected="true"]')).toHaveLength(1);

    await act(() => userEvent.type(input, 'Command'));
    await expectHighlighted('Command 00');
    input.setSelectionRange(3, 3);
    await act(() => userEvent.keyboard('{ArrowDown}{Home}'));
    expect(input.selectionStart).toBe(0);
    await act(() => userEvent.keyboard('{End}'));
    expect(input.selectionStart).toBe('Command'.length);
    await expectHighlighted('Command 01');
    // Alt does not hand them to the highlight either.
    await act(() => userEvent.keyboard('{Alt>}{Home}{/Alt}'));
    expect(input.selectionStart).toBe(0);
    await act(() => userEvent.keyboard('{Alt>}{End}{/Alt}'));
    expect(input.selectionStart).toBe('Command'.length);
    await expectHighlighted('Command 01');
  });

  it('highlights on hover and runs on click without taking focus from the search field', async () => {
    const betaRun = vi.fn();
    const { input, onClose } = await renderPalette({
      entries: [entry('alpha', 'Alpha'), entry('beta', 'Beta', betaRun)],
    });
    const beta = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find(
      (option) => option.textContent === 'Beta'
    )!;

    await act(() => userEvent.hover(beta));
    await waitFor(() => expect(input.getAttribute('aria-activedescendant')).toBe(beta.id));

    await act(() => userEvent.click(beta));
    expect(betaRun).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
    expect(document.activeElement).toBe(input);
  });

  it('leaves keys that compose IME text to the composition', async () => {
    const run = vi.fn();
    const { input, onClose } = await renderPalette({
      entries: [entry('first', 'First command', run), entry('second', 'Second command')],
    });
    const highlighted = () => document.getElementById(input.getAttribute('aria-activedescendant') ?? '')?.textContent;
    const composingKeyDown = (key: string, flag: 'isComposing' | 'keyCode229') => {
      const event = new KeyboardEvent('keydown', {
        bubbles: true,
        cancelable: true,
        isComposing: flag === 'isComposing',
        key,
      });

      if (flag === 'keyCode229') {
        Object.defineProperty(event, 'keyCode', { value: 229 });
      }
      input.dispatchEvent(event);
    };

    for (const flag of ['isComposing', 'keyCode229'] as const) {
      await act(() => composingKeyDown('PageDown', flag));
      expect(highlighted()).toBe('First command');
      await act(() => composingKeyDown('Enter', flag));
      expect(run).not.toHaveBeenCalled();
      await act(() => composingKeyDown('Escape', flag));
      expect(onClose).not.toHaveBeenCalled();
    }

    await act(() => userEvent.keyboard('{PageDown}'));
    expect(highlighted()).toBe('Second command');
  });

  it('runs the primary action on a click with Ctrl or ⌘ held, keeping the secondary on Mod+Enter', async () => {
    for (const modifier of ['Control', 'Meta'] as const) {
      // macOS reads a Ctrl+click as a context-menu click, which runs nothing.
      const clickRuns = IS_MAC_OS && modifier === 'Control' ? 0 : 1;
      const run = vi.fn();
      const secondaryRun = vi.fn();
      await renderPalette({
        entries: [
          { ...entry('first', 'First command', run), secondary: { label: 'Open Elsewhere', run: secondaryRun } },
        ],
      });
      const option = document.querySelector<HTMLElement>('[role="option"]')!;

      await act(() => userEvent.click(option, { modifiers: [modifier] }));
      expect(run).toHaveBeenCalledTimes(clickRuns);
      expect(secondaryRun).not.toHaveBeenCalled();

      await act(() => userEvent.keyboard(`{${modifier}>}{Enter}{/${modifier}}`));
      expect(secondaryRun).toHaveBeenCalledOnce();
      expect(run).toHaveBeenCalledTimes(clickRuns);

      await act(() => root?.unmount());
      host?.remove();
      root = null;
    }
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
