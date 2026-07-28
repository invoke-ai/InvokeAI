import type { WildcardCatalog } from '@features/generation/ui/useWildcards';

import { ChakraProvider } from '@chakra-ui/react';
import { WildcardsPanel } from '@features/generation/ui/promptFields/WildcardsPanel';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

// The panel reports delete failures through Generation's UI port, which only the
// app composes. Everything else in the module stays real.
vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({ notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn() } }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const catalog: WildcardCatalog = {
  create: vi.fn(),
  isLoading: false,
  knownNames: new Set(['colors']),
  remove: vi.fn(),
  update: vi.fn(),
  wildcards: [{ id: 'w1', name: 'colors', values: ['red', 'green'] }],
};

const renderPanel = async (showSyntaxHighlighting = true) => {
  host = document.createElement('div');
  host.style.width = '380px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <WildcardsPanel catalog={catalog} showSyntaxHighlighting={showSyntaxHighlighting} onInsert={vi.fn()} />
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
});

describe('wildcard values editor', () => {
  it('colours attention and nested dynamic syntax in the values', async () => {
    await openEditor(true);

    const spans = [...host!.querySelectorAll('[aria-hidden="true"] pre span')].map((span) => span.textContent);

    // Attention weighting survives into the generated prompt, and a nested
    // `{a|b}` is live syntax inside a wildcard value, so both are coloured.
    expect(spans).toContain('1.3');
    expect(spans).toContain('+');
    expect(spans).toContain('{');
    expect(spans).toContain('|');
  });

  it('numbers each value in the gutter', async () => {
    await openEditor(true);

    expect([...host!.querySelectorAll('[data-line-number]')].map((entry) => entry.textContent)).toEqual([
      '1',
      '2',
      '3',
    ]);
  });

  it('still numbers the values when highlighting is off', async () => {
    // Line numbers are not syntax colouring, so they do not follow that setting.
    await openEditor(false);

    expect([...host!.querySelectorAll('[data-line-number]')]).toHaveLength(3);
    expect(host!.querySelectorAll('[aria-hidden="true"] pre span')).toHaveLength(0);
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

  it('cannot save an empty name, and says nothing about it', async () => {
    await startCreate();

    expect(saveButton().disabled).toBe(true);
    expect(host!.querySelector('[role="alert"]')).toBeNull();
  });

  it('explains an unusable name before any request is made', async () => {
    await startCreate();
    await fillName('colors list');

    expect(host!.querySelector('[role="alert"]')?.textContent).toContain('wildcardNameInvalid');
    expect(saveButton().disabled).toBe(true);
    expect(catalog.create).not.toHaveBeenCalled();
  });

  it('flags a name the user already owns', async () => {
    await startCreate();
    await fillName('colors');

    expect(host!.querySelector('[role="alert"]')?.textContent).toContain('wildcardNameTaken');
  });

  it('accepts a nested name', async () => {
    await startCreate();
    await fillName('animals/dogs');

    expect(host!.querySelector('[role="alert"]')).toBeNull();
    expect(saveButton().disabled).toBe(false);
  });
});

describe('deleting a wildcard', () => {
  const clickDelete = async () => {
    // Insert, edit, delete — the trash is the last control on the row.
    const buttons = [...host!.querySelectorAll('button')];

    await act(async () => {
      await userEvent.click(buttons.at(-1)!);
    });
  };
  const dialogButton = (label: string) =>
    [...document.querySelectorAll<HTMLElement>('[role="alertdialog"] button')].find(
      (button) => button.textContent === label
    );

  // Regression: the trash icon deleted a wildcard and all of its values outright.
  it('asks before deleting', async () => {
    await renderPanel();
    await clickDelete();

    expect(document.querySelector('[role="alertdialog"]')).toBeTruthy();
    expect(catalog.remove).not.toHaveBeenCalled();
  });

  it('leaves the wildcard alone when the dialog is cancelled', async () => {
    await renderPanel();
    await clickDelete();

    await act(async () => {
      await userEvent.click(dialogButton('Cancel')!);
    });

    expect(catalog.remove).not.toHaveBeenCalled();
  });

  it('deletes once confirmed', async () => {
    await renderPanel();
    await clickDelete();

    await act(async () => {
      await userEvent.click(dialogButton('common.delete')!);
    });

    expect(catalog.remove).toHaveBeenCalledWith('w1');
  });
});
