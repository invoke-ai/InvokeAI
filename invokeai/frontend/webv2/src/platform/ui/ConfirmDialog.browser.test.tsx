import { ChakraProvider, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useCallback, useState, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ConfirmDialog } from './ConfirmDialog';
import { closingFrames, recordDialogExit } from './dialogExit.testing';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const noop = () => undefined;
// A control in the body, which a dialog left to itself might focus first.
const OPTION_BODY = (
  <label>
    <input type="checkbox" /> Option
  </label>
);
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

  it('is described by a text body, which looks as the plain paragraph it was before', async () => {
    // Long enough to wrap in the dialog, so a line height of its own shows in the box too.
    const BODY =
      'Queued runs are canceled, and all of this calibration’s measurements are discarded, finished runs included.';
    /** Opens a dialog with `body` and reads how the paragraph holding BODY renders. */
    const readParagraph = async (body: ReactNode) => {
      host = document.createElement('div');
      document.body.append(host);
      root = createRoot(host);
      await act(() =>
        root?.render(
          <ChakraProvider value={system}>
            <ConfirmDialog body={body} confirmLabel="Stop" isOpen title="Stop?" onClose={noop} onConfirm={noop} />
          </ChakraProvider>
        )
      );
      await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');
      const dialog = document.querySelector('[role="alertdialog"]')!;
      const paragraph = [...dialog.querySelectorAll('*')].find(
        (element) => element.childElementCount === 0 && element.textContent === BODY
      )!;
      const style = getComputedStyle(paragraph);
      const read = {
        describes: dialog.getAttribute('aria-describedby') === paragraph.id,
        style: {
          color: style.color,
          fontSize: style.fontSize,
          fontWeight: style.fontWeight,
          height: paragraph.getBoundingClientRect().height,
          letterSpacing: style.letterSpacing,
          lineHeight: style.lineHeight,
          margin: style.margin,
        },
      };

      await act(() => root?.unmount());
      host.remove();
      host = null;
      root = null;
      return read;
    };

    const described = await readParagraph(BODY);
    // What a text body rendered before it described the dialog.
    const plain = await readParagraph(<Text fontSize="md">{BODY}</Text>);

    expect(described.describes).toBe(true);
    expect(plain.describes).toBe(false);
    expect(described.style).toEqual(plain.style);
    // It wraps, so the comparison covers the lines' spacing.
    expect(described.style.height).toBeGreaterThan(Number.parseFloat(described.style.lineHeight) * 1.5);
  });

  const renderDialog = async (isDestructive: boolean) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <ConfirmDialog
            body={OPTION_BODY}
            confirmLabel="Go"
            isDestructive={isDestructive}
            isOpen
            title="Sure?"
            onClose={noop}
            onConfirm={noop}
          />
        </ChakraProvider>
      )
    );
    await expect
      .poll(() => document.querySelector('[role="alertdialog"]')?.contains(document.activeElement), { timeout: 5000 })
      .toBe(true);
  };
  const focusedLabel = () => (document.activeElement as HTMLElement | null)?.textContent?.trim() ?? '';

  it('opens a destructive confirmation on Cancel, ahead of a control in its body', async () => {
    await renderDialog(true);

    expect(focusedLabel()).toBe('Cancel');
  });

  it('leaves a non-destructive confirmation on the dialog’s own initial focus', async () => {
    await renderDialog(false);

    expect(document.activeElement).not.toBe(
      [...document.querySelectorAll('button')].find((button) => button.textContent === 'Cancel')
    );
  });
});
