import type { WildcardImportEntry } from '@features/generation/core/wildcardTransfer';

import { ChakraProvider } from '@chakra-ui/react';
import { WildcardImportDialog } from '@features/generation/ui/promptFields/WildcardImportDialog';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

let root: Root | null = null;
let host: HTMLDivElement | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const entry = (overrides: Partial<WildcardImportEntry> & { name: string }): WildcardImportEntry => ({
  conflictId: null,
  rejection: null,
  values: ['a'],
  ...overrides,
});

const ENTRIES: WildcardImportEntry[] = [
  entry({ name: 'fresh' }),
  entry({ conflictId: 'w1', name: 'colours', values: ['red', 'green'] }),
  entry({ conflictId: 'w2', name: 'moods' }),
  entry({ name: 'bad name!', rejection: 'invalid' }),
];

const renderDialog = async (entries = ENTRIES) => {
  const onConfirm = vi.fn(() => Promise.resolve());

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <WildcardImportDialog entries={entries} onCancel={vi.fn()} onConfirm={onConfirm} />
      </ChakraProvider>
    );
  });

  return onConfirm;
};

const dialog = () => document.querySelector('[role="alertdialog"]')!;
const segmentGroups = () => [
  ...dialog().querySelectorAll<HTMLElement>('[data-scope="segment-group"][data-part="root"]'),
];
const clickOption = async (group: HTMLElement, label: string) => {
  const option = [...group.querySelectorAll<HTMLElement>('label')].find((item) => item.textContent?.includes(label));

  await act(async () => {
    await userEvent.click(option!);
  });
};
const confirm = async () => {
  const button = [...dialog().querySelectorAll('button')].find((item) => item.textContent?.includes('import'));

  await act(async () => {
    await userEvent.click(button!);
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('wildcard import dialog', () => {
  it('says how much is new and how much already exists', async () => {
    await renderDialog();

    expect(dialog().textContent).toContain('importSummary');
  });

  // Import is the one action here that can overwrite a hand-typed list.
  it('leaves every clash alone unless told otherwise', async () => {
    const onConfirm = await renderDialog();

    await confirm();

    expect(onConfirm).toHaveBeenCalledWith({});
  });

  it('resolves one clash without touching the other', async () => {
    const onConfirm = await renderDialog();
    // The first group is the apply-to-all row; the rest are the clashes.
    const [, colours] = segmentGroups();

    await clickOption(colours, 'importReplace');
    await confirm();

    expect(onConfirm).toHaveBeenCalledWith({ colours: 'replace' });
  });

  it('applies one decision to every clash', async () => {
    const onConfirm = await renderDialog();
    const [all] = segmentGroups();

    await clickOption(all, 'importKeepBoth');
    await confirm();

    expect(onConfirm).toHaveBeenCalledWith({ colours: 'keepBoth', moods: 'keepBoth' });
  });

  it('lets a row override the decision made for all of them', async () => {
    const onConfirm = await renderDialog();
    const [all, colours] = segmentGroups();

    await clickOption(all, 'importReplace');
    await clickOption(colours, 'importSkip');
    await confirm();

    expect(onConfirm).toHaveBeenCalledWith({ colours: 'skip', moods: 'replace' });
  });

  // A count that said "3 imported" while quietly dropping the fourth would be
  // the worst version of this dialog.
  it('names what it cannot import, and why', async () => {
    await renderDialog();

    expect(dialog().textContent).toContain('bad name!');
    expect(dialog().textContent).toContain('importRejectedInvalid');
  });

  it('offers no apply-to-all for a single clash', async () => {
    await renderDialog([entry({ conflictId: 'w1', name: 'colours' })]);

    expect(segmentGroups()).toHaveLength(1);
  });
});
