import type { WildcardCatalog } from '@features/generation/ui/useWildcards';

import { ChakraProvider } from '@chakra-ui/react';
import { WildcardsPanel } from '@features/generation/ui/promptFields/WildcardsPanel';
import { WILDCARD_COLLECTION_FORMATS } from '@features/generation/ui/wildcardFiles';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const { reportError } = vi.hoisted(() => ({ reportError: vi.fn() }));

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({ notifications: { error: vi.fn(), info: vi.fn(), reportError } }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const catalog: WildcardCatalog = {
  applyWrites: vi.fn(),
  create: vi.fn(),
  errorDetail: null,
  isError: false,
  isLoading: false,
  isRetrying: false,
  knownNames: new Set(['colors']),
  remove: vi.fn(),
  removeMany: vi.fn(),
  retry: vi.fn(),
  update: vi.fn(),
  wildcards: [{ id: 'w1', name: 'colors', values: ['red', 'green'] }],
};

const emptyCatalog: WildcardCatalog = { ...catalog, wildcards: [] };

const renderPanel = async (
  showSyntaxHighlighting = true,
  panelCatalog: WildcardCatalog = catalog,
  onInsert = vi.fn()
) => {
  host = document.createElement('div');
  host.style.width = '380px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <WildcardsPanel catalog={panelCatalog} showSyntaxHighlighting={showSyntaxHighlighting} onInsert={onInsert} />
      </ChakraProvider>
    );
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  accountLifecycle.invalidate();
  vi.restoreAllMocks();
});

it('does not automatically write a YAML import whose parser finishes after account rotation', async () => {
  let resolveParse!: (value: unknown) => void;
  const parsed = new Promise<unknown>((resolve) => {
    resolveParse = resolve;
  });
  const yaml = WILDCARD_COLLECTION_FORMATS.find((format) => format.id === 'yaml')!;
  const parse = vi.spyOn(yaml, 'parse').mockReturnValue(parsed);
  const applyWrites = vi.fn(() => Promise.resolve(1));

  accountLifecycle.activate('wildcard-import-a', ':user:wildcard-import-a');
  await renderPanel(true, { ...catalog, applyWrites, knownNames: new Set(), wildcards: [] });

  const input = host!.querySelector<HTMLInputElement>(
    `input[accept="${'.txt,.yaml,.yml,.json,text/plain,application/json'}"]`
  )!;
  await act(async () => {
    await userEvent.upload(input, new File(['colors:\\n  - red\\n'], 'wildcards.yaml', { type: 'text/yaml' }));
    await vi.waitFor(() => expect(parse).toHaveBeenCalledOnce());
  });

  await act(async () => {
    accountLifecycle.activate('wildcard-import-b', ':user:wildcard-import-b');
    resolveParse({ colors: ['red'] });
    await parsed;
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

  expect(applyWrites).not.toHaveBeenCalled();
});

it('animates the conflict dialog out instead of unmounting it on cancel', async () => {
  accountLifecycle.activate('wildcard-dialog-exit', ':user:wildcard-dialog-exit');
  await renderPanel();

  const input = host!.querySelector<HTMLInputElement>(
    `input[accept="${'.txt,.yaml,.yml,.json,text/plain,application/json'}"]`
  )!;
  await act(async () => {
    await userEvent.upload(input, new File(['blue'], 'colors.txt', { type: 'text/plain' }));
  });
  await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');
  const dialog = document.querySelector('[role="alertdialog"]')!;

  const frames = closingFrames(await recordDialogExit(dialog, () => act(() => userEvent.keyboard('{Escape}'))));
  await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();

  expect(frames).not.toHaveLength(0);
  for (const frame of frames) {
    expect(frame.text).toContain('colors');
  }
});

describe('a failed wildcard read', () => {
  it('says the read failed and why, with a retry, instead of offering to create the first wildcard', async () => {
    const failed = { ...emptyCatalog, errorDetail: 'Database is locked', isError: true, retry: vi.fn() };

    await renderPanel(true, failed);

    expect(host!.querySelector('[role="alert"]')?.textContent).toContain(
      'widgets.generate.dynamicPrompts.couldNotLoadWildcards'
    );
    expect(host!.querySelector('[role="alert"]')?.textContent).toContain('Database is locked');
    expect(host!.textContent).not.toContain('widgets.generate.dynamicPrompts.noWildcardsYet');
    expect(host!.innerHTML).not.toContain('widgets.generate.dynamicPrompts.selectAllWildcards');
    expect(host!.textContent).not.toContain('widgets.generate.dynamicPrompts.export');

    await act(async () => {
      await userEvent.click(
        [...host!.querySelectorAll('button')].find((button) => button.textContent === 'common.retry')!
      );
    });

    expect(failed.retry).toHaveBeenCalledOnce();
  });
});

describe('deleting a wildcard', () => {
  const clickDelete = async () => {
    const trash = host!.querySelector<HTMLButtonElement>('button[aria-label="common.delete"]');

    await act(async () => {
      await userEvent.click(trash!);
    });
  };
  const dialogButton = (label: string) =>
    [...document.querySelectorAll<HTMLElement>('[role="alertdialog"] button')].find(
      (button) => button.textContent === label
    );

  it('deletes once confirmed', async () => {
    await renderPanel();
    await clickDelete();

    await act(async () => {
      await userEvent.click(dialogButton('common.delete')!);
    });

    expect(catalog.remove).toHaveBeenCalledWith('w1');
  });
});

describe('selecting several wildcards', () => {
  const wildcards = [
    { id: 'w1', name: 'colors', values: ['red', 'green'] },
    { id: 'w2', name: 'animals/dogs', values: ['corgi'] },
    { id: 'w3', name: 'animals/cats', values: ['tabby'] },
    { id: 'w4', name: 'moods', values: ['serene'] },
  ];
  const selectingCatalog = (removeMany: WildcardCatalog['removeMany'] = vi.fn(() => Promise.resolve([]))) => ({
    ...catalog,
    knownNames: new Set(wildcards.map((wildcard) => wildcard.name)),
    removeMany,
    wildcards,
  });
  const rowCheckbox = (id: string) =>
    host!.querySelector<HTMLInputElement>(`[data-list-row="${id}"] input[type="checkbox"]`)!;
  const selectAll = () =>
    host!.querySelector<HTMLInputElement>(
      '[aria-label="widgets.generate.dynamicPrompts.selectAllWildcards"] input[type="checkbox"]'
    )!;
  const checkedIds = () => wildcards.map((wildcard) => wildcard.id).filter((id) => rowCheckbox(id)?.checked);
  const toggle = async (input: HTMLInputElement) => {
    await act(() => input.click());
  };
  const button = (label: string, scope: ParentNode = host!) =>
    [...scope.querySelectorAll<HTMLButtonElement>('button')].find(
      (candidate) => candidate.textContent === label || candidate.getAttribute('aria-label') === label
    );
  const confirmButton = () => button('widgets.generate.dynamicPrompts.bulkDeleteWildcardsConfirm', document);

  it('deletes every selected wildcard after one confirm', async () => {
    const removeMany = vi.fn((ids: readonly string[]) => Promise.resolve([...ids]));
    await renderPanel(true, selectingCatalog(removeMany));

    await toggle(selectAll());
    await act(async () => {
      await userEvent.click(
        button(
          'common.delete',
          host!.querySelector('[aria-label="widgets.generate.dynamicPrompts.selectAllWildcards"]')!.parentElement!
        )!
      );
    });
    await act(async () => {
      await userEvent.click(confirmButton()!);
    });

    expect(removeMany).toHaveBeenCalledOnce();
    expect([...removeMany.mock.calls[0]![0]].sort()).toEqual(['w1', 'w2', 'w3', 'w4']);
    expect(checkedIds()).toEqual([]);
  });

  it('returns focus into the panel after deleting every wildcard, not to the page', async () => {
    // As the real catalog does, the refetch that follows the delete lands before it resolves: nothing is left to list,
    // select or search.
    const removeMany = vi.fn((ids: readonly string[]) => {
      root?.render(
        <ChakraProvider value={system}>
          <WildcardsPanel catalog={emptyCatalog} showSyntaxHighlighting onInsert={vi.fn()} />
        </ChakraProvider>
      );
      return Promise.resolve([...ids]);
    });
    await renderPanel(true, selectingCatalog(removeMany));

    await toggle(selectAll());
    await act(async () => {
      await userEvent.click(
        button(
          'common.delete',
          host!.querySelector('[aria-label="widgets.generate.dynamicPrompts.selectAllWildcards"]')!.parentElement!
        )!
      );
    });
    await act(async () => {
      await userEvent.click(confirmButton()!);
    });

    await vi.waitFor(() => expect(document.activeElement).toBe(button('widgets.generate.dynamicPrompts.newWildcard')));
  });
});
