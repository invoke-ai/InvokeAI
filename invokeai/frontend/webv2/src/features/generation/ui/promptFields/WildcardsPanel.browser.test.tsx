import type { WildcardCatalog } from '@features/generation/ui/useWildcards';

import { ChakraProvider, Field, Input } from '@chakra-ui/react';
import { WildcardsPanel } from '@features/generation/ui/promptFields/WildcardsPanel';
import { WildcardDeleteError } from '@features/generation/ui/useWildcards';
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

/** The catalog once everything is deleted. */
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

const openEditor = async (showSyntaxHighlighting: boolean) => {
  await renderPanel(showSyntaxHighlighting);

  await act(async () => {
    await userEvent.click([...host!.querySelectorAll('button')][0]!);
  });

  const values = host!.querySelector('textarea')!;

  await act(async () => {
    await userEvent.fill(values, '(oil painting)1.3\nwatercolor+\n{ink|charcoal} sketch');
  });
  await act(async () => {
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
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

it('quietly refuses a conflict-dialog import after its originating account rotates', async () => {
  const applyWrites = vi.fn(() => Promise.resolve(1));

  accountLifecycle.activate('wildcard-dialog-a', ':user:wildcard-dialog-a');
  await renderPanel(true, { ...catalog, applyWrites });

  const input = host!.querySelector<HTMLInputElement>(
    `input[accept="${'.txt,.yaml,.yml,.json,text/plain,application/json'}"]`
  )!;
  await act(async () => {
    await userEvent.upload(input, new File(['blue'], 'colors.txt', { type: 'text/plain' }));
  });
  await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).not.toBeNull());

  accountLifecycle.activate('wildcard-dialog-b', ':user:wildcard-dialog-b');
  const confirm = [...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')].find((button) =>
    button.textContent?.includes('import')
  )!;

  await act(async () => {
    await userEvent.click(confirm);
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

describe('wildcard list size', () => {
  const viewportHeight = () => host!.querySelector('[data-list-viewport]')!.getBoundingClientRect().height;

  it('fits a short list to its rows instead of leaving an empty box', async () => {
    await renderPanel();

    const row = host!.querySelector<HTMLElement>('[data-list-row="w1"]')!;
    const slot = row.parentElement!.getBoundingClientRect().height;

    const description = [...row.querySelectorAll('span')].find((span) => span.textContent === 'red, green')!;

    expect(viewportHeight()).toBe(slot);
    // The value line sits inside the row's fixed slot, which would otherwise clip it.
    expect(description.getBoundingClientRect().bottom).toBeLessThanOrEqual(row.getBoundingClientRect().bottom);
  });

  it('caps a long list and scrolls it', async () => {
    await renderPanel(true, {
      ...catalog,
      wildcards: Array.from({ length: 20 }, (_, index) => ({ id: `w${index}`, name: `set${index}`, values: ['a'] })),
    });

    expect(viewportHeight()).toBe(14 * 16);
  });

  it('keeps its height while a search narrows the matches', async () => {
    await renderPanel(true, {
      ...catalog,
      wildcards: [
        { id: 'w1', name: 'colors', values: ['red'] },
        { id: 'w2', name: 'moods', values: ['serene'] },
      ],
    });
    const before = viewportHeight();
    const search = host!.querySelector<HTMLInputElement>('input[aria-label$="searchWildcards"]')!;

    await act(async () => {
      await userEvent.fill(search, 'colors');
    });

    expect(viewportHeight()).toBe(before);
  });
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
    // Selecting and exporting act on a list there is none of.
    expect(host!.innerHTML).not.toContain('widgets.generate.dynamicPrompts.selectAllWildcards');
    expect(host!.textContent).not.toContain('widgets.generate.dynamicPrompts.export');

    await act(async () => {
      await userEvent.click(
        [...host!.querySelectorAll('button')].find((button) => button.textContent === 'common.retry')!
      );
    });

    expect(failed.retry).toHaveBeenCalledOnce();
  });

  it('keeps the wildcards it has when a refresh fails', async () => {
    await renderPanel(true, { ...catalog, isError: true });

    expect(host!.textContent).toContain('widgets.generate.dynamicPrompts.couldNotRefreshWildcards');
    expect(host!.textContent).toContain('__colors__');
  });

  it('says it is loading rather than empty before the first read', async () => {
    await renderPanel(true, { ...emptyCatalog, isLoading: true });

    expect(host!.textContent).toContain('widgets.generate.dynamicPrompts.loadingWildcards');
    expect(host!.textContent).not.toContain('widgets.generate.dynamicPrompts.noWildcardsYet');
  });
});

describe('wildcard values editor', () => {
  it('colours attention and nested dynamic syntax in the values', async () => {
    await openEditor(true);

    const spans = [...host!.querySelectorAll('[aria-hidden="true"] pre span')].map((span) => span.textContent);

    expect(spans).toContain('1.3');
    expect(spans).toContain('+');
    expect(spans).toContain('{');
    expect(spans).toContain('|');
  });
});

describe('wildcard name validation', () => {
  const nameInput = () => host!.querySelector('input')!;
  const saveButton = () => [...host!.querySelectorAll('button')].find((b) => b.textContent?.includes('common.save'))!;
  const fillName = async (name: string) => {
    await act(async () => {
      await userEvent.fill(nameInput(), name);
    });
  };

  const startCreate = async () => {
    await renderPanel();
    await act(async () => {
      await userEvent.click([...host!.querySelectorAll('button')][0]!);
    });
  };

  it('explains an unusable name before any request is made', async () => {
    await startCreate();
    await fillName('colors list');

    expect(host!.querySelector('[role="alert"]')?.textContent).toContain('wildcardNameInvalid');
    expect(saveButton().disabled).toBe(true);
    expect(catalog.create).not.toHaveBeenCalled();
  });
});

describe('finding a wildcard', () => {
  const manyCatalog: WildcardCatalog = {
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
    wildcards: [
      { id: 'w1', name: 'colors', values: ['red', 'green'] },
      { id: 'w2', name: 'animals/dogs', values: ['corgi', 'husky'] },
      { id: 'w3', name: 'animals/cats', values: ['tabby'] },
      { id: 'w4', name: 'moods', values: ['cyberpunk', 'serene'] },
    ],
  };

  const searchInput = () =>
    [...host!.querySelectorAll('input')].find((input) => input.getAttribute('aria-label')?.includes('searchWildcards'));
  const referenceLabels = () =>
    [...host!.querySelectorAll('span')].map((span) => span.textContent).filter((text) => text?.startsWith('__'));
  it('offers no search box until there is something to search', async () => {
    await renderPanel(true, { ...manyCatalog, wildcards: [] });

    expect(searchInput()).toBeUndefined();
    expect(host!.textContent).toContain('noWildcardsYet');
  });

  it('heads nested names with their shared prefix', async () => {
    await renderPanel(true, manyCatalog);

    expect([...host!.querySelectorAll('p, span')].map((node) => node.textContent)).toContain('animals');
  });

  it('still shows the whole reference under a group header', async () => {
    await renderPanel(true, manyCatalog);

    expect(referenceLabels()).toContain('__animals/dogs__');
    expect(referenceLabels()).not.toContain('__dogs__');
  });
});

describe('wildcard list interactions', () => {
  const selectableWildcardButton = () =>
    host!.querySelector<HTMLButtonElement>('[data-list-row="w1"] [data-list-primary]')!;

  it('keeps the edit controls outside the selectable row button', async () => {
    const onInsert = vi.fn();
    await renderPanel(true, catalog, onInsert);

    const row = selectableWildcardButton();
    const edit = host!.querySelector<HTMLButtonElement>('button[aria-label="common.edit"]')!;
    const remove = host!.querySelector<HTMLButtonElement>('button[aria-label="common.delete"]')!;

    expect(row.contains(edit)).toBe(false);
    expect(row.contains(remove)).toBe(false);
    // The row's name says what pressing it does, not only which wildcard it is.
    expect(row.getAttribute('aria-label')).toBe('widgets.generate.dynamicPrompts.insertWildcard: __colors__');

    await act(async () => {
      await userEvent.click(row);
    });

    expect(onInsert).toHaveBeenCalledWith('__colors__');
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
  const selectAllState = () =>
    host!
      .querySelector('[aria-label="widgets.generate.dynamicPrompts.selectAllWildcards"]')!
      .getAttribute('data-state');
  const checkedIds = () => wildcards.map((wildcard) => wildcard.id).filter((id) => rowCheckbox(id)?.checked);
  const toggle = async (input: HTMLInputElement) => {
    await act(() => input.click());
  };
  const search = async (term: string) => {
    const input = host!.querySelector<HTMLInputElement>(
      'input[aria-label="widgets.generate.dynamicPrompts.searchWildcards"]'
    )!;

    await act(async () => {
      await userEvent.fill(input, term);
    });
  };
  const button = (label: string, scope: ParentNode = host!) =>
    [...scope.querySelectorAll<HTMLButtonElement>('button')].find(
      (candidate) => candidate.textContent === label || candidate.getAttribute('aria-label') === label
    );
  const confirmButton = () => button('widgets.generate.dynamicPrompts.bulkDeleteWildcardsConfirm', document);

  it('gives every row a checkbox while a press on the row still inserts it', async () => {
    const onInsert = vi.fn();
    await renderPanel(true, selectingCatalog(), onInsert);

    expect(wildcards.every((wildcard) => rowCheckbox(wildcard.id) !== null)).toBe(true);
    expect(button('common.delete')).not.toBeUndefined(); // the row's own trash action
    expect(button('widgets.generate.dynamicPrompts.clearWildcardSelection')).toBeUndefined();

    await act(async () => {
      await userEvent.click(host!.querySelector<HTMLElement>('[data-list-row="w4"] [data-list-primary]')!);
    });

    expect(onInsert).toHaveBeenCalledWith('__moods__');
    expect(checkedIds()).toEqual([]);
  });

  it('selects all of the filtered list, adding to what an earlier search selected', async () => {
    await renderPanel(true, selectingCatalog());

    await toggle(rowCheckbox('w4'));
    expect(selectAllState()).toBe('indeterminate');

    await search('animals');
    await toggle(selectAll());
    expect(selectAllState()).toBe('checked');

    await search('');
    expect(checkedIds()).toEqual(['w2', 'w3', 'w4']);
    expect(selectAllState()).toBe('indeterminate');
  });

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

  it('keeps what could not be deleted selected and reports how many were', async () => {
    const removeMany = vi.fn(() => Promise.reject(new WildcardDeleteError(['w1'], new Error('offline'))));
    await renderPanel(true, selectingCatalog(removeMany));

    await toggle(rowCheckbox('w1'));
    await toggle(rowCheckbox('w2'));
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

    expect(checkedIds()).toEqual(['w2']);
    expect(reportError).toHaveBeenCalledWith(expect.objectContaining({ area: 'delete-wildcard' }));
  });

  it('clears the selection', async () => {
    await renderPanel(true, selectingCatalog());

    await toggle(selectAll());
    await act(async () => {
      await userEvent.click(button('widgets.generate.dynamicPrompts.clearWildcardSelection')!);
    });

    expect(checkedIds()).toEqual([]);
    expect(selectAllState()).toBe('unchecked');
    // The Clear selection button is gone; focus stays in the panel, on select all.
    expect(document.activeElement).toBe(selectAll());
  });

  it('moves focus to the search field when clearing a selection the search hides', async () => {
    await renderPanel(true, selectingCatalog());

    await toggle(rowCheckbox('w1'));
    await search('no such wildcard');
    await act(async () => {
      await userEvent.click(button('widgets.generate.dynamicPrompts.clearWildcardSelection')!);
    });

    expect(document.activeElement).toBe(
      host!.querySelector('input[aria-label="widgets.generate.dynamicPrompts.searchWildcards"]')
    );
  });

  it('returns focus into the panel after deleting every wildcard, not to the page', async () => {
    // As the real catalog does, the refetch that follows the delete lands before it resolves: nothing is left to
    // list, select or search.
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

  it('selects a row from the keyboard', async () => {
    await renderPanel(true, selectingCatalog());

    rowCheckbox('w1').focus();
    await act(async () => {
      await userEvent.keyboard(' ');
    });

    expect(checkedIds()).toEqual(['w1']);
  });
});

it('gives its search box a field of its own inside the prompt field it opens from', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Field.Root invalid>
          <Input aria-label="Prompt" />
          <WildcardsPanel catalog={catalog} showSyntaxHighlighting onInsert={vi.fn()} />
        </Field.Root>
      </ChakraProvider>
    );
  });

  const prompt = host.querySelector<HTMLInputElement>('input[aria-label="Prompt"]')!;
  const search = host.querySelector<HTMLInputElement>(
    'input[aria-label="widgets.generate.dynamicPrompts.searchWildcards"]'
  )!;

  expect(search.id).not.toBe(prompt.id);
  expect(search.getAttribute('aria-invalid')).toBeNull();
});
