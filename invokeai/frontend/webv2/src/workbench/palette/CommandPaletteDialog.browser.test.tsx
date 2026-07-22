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
