import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ConfirmDialog } from './ConfirmDialog';
import { closingFrames, recordDialogExit } from './dialogExit.testing';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const noop = () => undefined;
let confirmCalls = 0;
const handleConfirm = () => {
  confirmCalls += 1;
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ConfirmDialog', () => {
  it('keeps confirmation disabled until its owner has a valid preview', async () => {
    confirmCalls = 0;
    const renderDialog = async (isConfirmDisabled: boolean) => {
      await act(() =>
        root?.render(
          <ChakraProvider value={system}>
            <ConfirmDialog
              body="Preview is still loading."
              confirmLabel="Proceed"
              isConfirmDisabled={isConfirmDisabled}
              isOpen
              title="Confirm maintenance"
              onClose={noop}
              onConfirm={handleConfirm}
            />
          </ChakraProvider>
        )
      );
    };

    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await renderDialog(true);

    const confirm = [...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')].find(
      (button) => button.textContent === 'Proceed'
    )!;
    expect(confirm.disabled).toBe(true);
    expect(confirmCalls).toBe(0);

    await renderDialog(false);
    const enabledConfirm = [...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')].find(
      (button) => button.textContent === 'Proceed'
    )!;
    expect(enabledConfirm.disabled).toBe(false);
    await act(() => userEvent.click(enabledConfirm));
    expect(confirmCalls).toBe(1);
  });

  it('keeps the text it showed while it animates out after its host clears the subject', async () => {
    // The usual host shape: the subject drives the copy and is cleared the moment the dialog closes.
    const Host = () => {
      const [subject, setSubject] = useState<string | null>('Sunset');
      const close = useCallback(() => setSubject(null), []);

      return (
        <ChakraProvider value={system}>
          <ConfirmDialog
            body={`Delete ${subject ?? ''}?`}
            confirmLabel={subject ? 'Delete' : 'OK'}
            isOpen={subject !== null}
            title={subject ? 'Delete image' : 'Delete items'}
            onClose={close}
            onConfirm={noop}
          />
        </ChakraProvider>
      );
    };
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => root?.render(<Host />));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');
    const dialog = document.querySelector('[role="alertdialog"]')!;
    const confirm = [...dialog.querySelectorAll('button')].find((button) => button.textContent === 'Delete')!;

    const frames = closingFrames(await recordDialogExit(dialog, () => act(() => userEvent.click(confirm))));

    expect(frames).not.toHaveLength(0);
    for (const frame of frames) {
      expect(frame.text).toContain('Delete image');
      expect(frame.text).toContain('Delete Sunset?');
    }
  });
});
