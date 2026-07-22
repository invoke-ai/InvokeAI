import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const entry = (id: string, title: string, run = vi.fn()): PaletteEntry => ({
  group: 'Commands',
  id,
  isPersistentRecent: true,
  run,
  showInEmptyState: true,
  title,
});

const renderPalette = async ({
  entries,
  onClose = vi.fn(),
  providers,
}: {
  entries: PaletteEntry[];
  onClose?: () => void;
  providers?: PaletteSearchProvider[];
}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
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
    );
  });

  const input = await vi.waitFor(() => {
    const node = document.querySelector<HTMLInputElement>('[name="command-palette-query"]');
    expect(node).not.toBeNull();
    return node!;
  });

  return { input, onClose };
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  window.localStorage.clear();
});

describe('CommandPaletteDialog interaction', () => {
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

    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 250);
    });
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
    const retry = await vi.waitFor(() => {
      const node = [...document.querySelectorAll('button')].find((button) => button.textContent === 'Retry');
      expect(node).toBeDefined();
      return node!;
    });

    await act(() => userEvent.tab());
    expect(document.activeElement).toBe(retry);

    await act(() => userEvent.click(retry));
    await vi.waitFor(() => expect(document.activeElement).toBe(input));
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
    await vi.waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith({
        range: { from: expect.stringMatching(/^\d{4}-\d{2}-\d{2}$/), to: undefined },
        text: '',
      });
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
    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 250);
    });

    await vi.waitFor(() => expect(capableSearch).toHaveBeenCalled());
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
    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 250);
    });

    // No acknowledgment chip, and the provider receives the literal text.
    expect(document.getElementById('command-palette-date-hint')).toBeNull();
    await vi.waitFor(() => {
      expect(dateLessSearch).toHaveBeenCalledWith({ range: undefined, text: 'from:7d' });
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
    await vi.waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith({ range: undefined, text: '' });
    });
    expect(document.body.textContent).toContain('Entity One');

    // The same command works from > commands mode and exits it into the scope.
    await act(() => userEvent.keyboard('{Backspace}'));
    await act(() => userEvent.fill(input, '>search ent'));
    await act(() => userEvent.keyboard('{Enter}'));
    expect(input.value).toBe('');
    await vi.waitFor(() => expect(document.body.textContent).toContain('Entity One'));
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
    await vi.waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith({ range: undefined, text: 'zz' });
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
    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 250);
    });
    expect(providerSearch).not.toHaveBeenCalled();

    await act(() => userEvent.fill(input, ''));
    await act(() => userEvent.keyboard('{Enter}'));
    await vi.waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith({ range: undefined, text: '' });
    });

    await act(() => userEvent.fill(input, 'a'));
    await vi.waitFor(() => {
      expect(providerSearch).toHaveBeenCalledWith({ range: undefined, text: 'a' });
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
    await vi.waitFor(() => expect(document.body.textContent).toContain('No entities yet'));

    await act(() => userEvent.fill(input, 'x'));
    await vi.waitFor(() => expect(document.body.textContent).toContain('No results for “x”'));
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
    await vi.waitFor(() => expect(document.body.textContent).toContain("Couldn't search entities"));
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
    // Deliberately listed twice: once under Recent, once under Search in.
    expect(document.body.textContent?.match(/Search entities…/g)).toHaveLength(2);
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
