import type { WildcardCatalog } from '@features/generation/ui/useWildcards';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { expectRowInteractionsToMatch } from '@features/generation/ui/promptFields/promptFieldsBrowserTestUtils';
import { WildcardsPanel } from '@features/generation/ui/promptFields/WildcardsPanel';
import { WILDCARD_COLLECTION_FORMATS } from '@features/generation/ui/wildcardFiles';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { Row } from '@platform/ui/Row';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({ notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn() } }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const catalog: WildcardCatalog = {
  applyWrites: vi.fn(),
  create: vi.fn(),
  isLoading: false,
  knownNames: new Set(['colors']),
  remove: vi.fn(),
  update: vi.fn(),
  wildcards: [{ id: 'w1', name: 'colors', values: ['red', 'green'] }],
};

const renderPanel = async (
  showSyntaxHighlighting = true,
  panelCatalog: WildcardCatalog = catalog,
  onInsert = vi.fn(),
  includeRowProbe = false
) => {
  host = document.createElement('div');
  host.style.width = '380px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        {includeRowProbe ? (
          <>
            <Box aria-hidden bg="bg.emphasized" data-testid="row-hover-style-probe" />
            <Row asChild>
              <button aria-label="Row probe" type="button">
                Row probe
              </button>
            </Row>
          </>
        ) : null}
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
    isLoading: false,
    knownNames: new Set(['colors']),
    remove: vi.fn(),
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
    host!.querySelector<HTMLButtonElement>('button[title="widgets.generate.dynamicPrompts.insertWildcard"]')!;

  it('uses the shared Row interaction contract without nesting the edit controls', async () => {
    const onInsert = vi.fn();
    await renderPanel(true, catalog, onInsert, true);

    const probe = host!.querySelector<HTMLButtonElement>('button[aria-label="Row probe"]')!;
    const hoverBackgroundColor = getComputedStyle(
      host!.querySelector('[data-testid="row-hover-style-probe"]')!
    ).backgroundColor;
    const row = selectableWildcardButton();
    const edit = host!.querySelector<HTMLButtonElement>('button[aria-label="common.edit"]')!;
    const remove = host!.querySelector<HTMLButtonElement>('button[aria-label="common.delete"]')!;

    expect(row.contains(edit)).toBe(false);
    expect(row.contains(remove)).toBe(false);

    await expectRowInteractionsToMatch(probe, row, hoverBackgroundColor);

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
