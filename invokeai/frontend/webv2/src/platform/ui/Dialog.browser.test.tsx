import { ChakraProvider, Portal } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, StrictMode, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { Dialog } from './Dialog';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const nextFrame = () =>
  new Promise<void>((resolve) => {
    requestAnimationFrame(() => resolve());
  });

/**
 * A dialog that opens a sibling dialog from inside itself, the way the workflow library opens its graph preview: the
 * child mounts already open and unmounts once it has animated out.
 */
const ParentWithChild = () => {
  const [isParentOpen, setIsParentOpen] = useState(true);
  const [isChildMounted, setIsChildMounted] = useState(false);
  const [isChildOpen, setIsChildOpen] = useState(false);
  const handleParentOpenChange = useCallback((details: Dialog.OpenChangeDetails) => setIsParentOpen(details.open), []);
  const openChild = useCallback(() => {
    setIsChildMounted(true);
    setIsChildOpen(true);
  }, []);
  const handleChildOpenChange = useCallback((details: Dialog.OpenChangeDetails) => setIsChildOpen(details.open), []);
  const closeChild = useCallback(() => setIsChildOpen(false), []);
  const unmountChild = useCallback(() => setIsChildMounted(false), []);

  return (
    <>
      <Dialog.Root open={isParentOpen} onOpenChange={handleParentOpenChange}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content aria-label="Parent">
              <Dialog.Body>
                <input aria-label="Search" />
                <button type="button" onClick={openChild}>
                  Open child
                </button>
              </Dialog.Body>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
      {isChildMounted ? (
        <Dialog.Root open={isChildOpen} onExitComplete={unmountChild} onOpenChange={handleChildOpenChange}>
          <Portal>
            <Dialog.Backdrop />
            <Dialog.Positioner>
              <Dialog.Content aria-label="Child">
                <Dialog.Body>
                  {/* The child's first tabbable, where a trap armed after focus entered the child would return it. */}
                  <button type="button">Graph</button>
                </Dialog.Body>
                <Dialog.Footer>
                  <button type="button" onClick={closeChild}>
                    Close
                  </button>
                </Dialog.Footer>
              </Dialog.Content>
            </Dialog.Positioner>
          </Portal>
        </Dialog.Root>
      ) : null}
    </>
  );
};

/** Closes on any dismissal, so a focus-outside dismissal that got through would close it. */
const ModalWithFocusOutside = ({ onFocusOutside }: { onFocusOutside: () => void }) => {
  const [isOpen, setIsOpen] = useState(true);
  const handleOpenChange = useCallback((details: Dialog.OpenChangeDetails) => setIsOpen(details.open), []);

  return (
    <Dialog.Root open={isOpen} onFocusOutside={onFocusOutside} onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Positioner>
          <Dialog.Content aria-label="Modal">
            <button type="button">Inside</button>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

const getDialog = (label: string) => document.querySelector<HTMLElement>(`[role="dialog"][aria-label="${label}"]`);
const getButton = (dialog: HTMLElement, text: string) =>
  [...dialog.querySelectorAll('button')].find((button) => button.textContent === text)!;

const renderParent = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <StrictMode>
        <ChakraProvider value={system}>
          <ParentWithChild />
        </ChakraProvider>
      </StrictMode>
    )
  );
  await expect.poll(() => getDialog('Parent')?.getAttribute('data-state')).toBe('open');

  return getDialog('Parent')!;
};

describe('Dialog', () => {
  it('stays open when a dialog it opened closes and focus returns to it', async () => {
    const parent = await renderParent();

    await act(() => userEvent.click(getButton(parent, 'Open child')));
    await expect.poll(() => getDialog('Child')?.getAttribute('data-state')).toBe('open');
    await act(() => userEvent.click(getButton(getDialog('Child')!, 'Close')));
    await expect.poll(() => getDialog('Child')).toBeNull();
    await expect.poll(() => document.activeElement).toBe(getButton(parent, 'Open child'));
    // Let the restored focus's outside handler run before checking that it did not dismiss the parent.
    await act(nextFrame);

    expect(parent.isConnected).toBe(true);
    expect(parent.getAttribute('data-state')).toBe('open');
    expect(document.activeElement).toBe(getButton(parent, 'Open child'));
  });

  it('still closes on a pointer press outside it', async () => {
    const parent = await renderParent();
    // The outside handling is set up in the frame after the dialog opens.
    await act(nextFrame);

    await act(() => userEvent.click(document.body, { position: { x: 4, y: 4 } }));

    await expect.poll(() => parent.getAttribute('data-state')).toBe('closed');
  });

  it('still tells the caller when focus lands outside a modal dialog, which stays open', async () => {
    const onFocusOutside = vi.fn();
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root!.render(
        <ChakraProvider value={system}>
          <button type="button">Outside</button>
          <ModalWithFocusOutside onFocusOutside={onFocusOutside} />
        </ChakraProvider>
      )
    );
    const dialog = getDialog('Modal')!;
    await expect.poll(() => dialog.getAttribute('data-state')).toBe('open');
    // The outside handling is set up in the frame after the dialog opens.
    await act(nextFrame);

    await act(() => host!.querySelector('button')!.focus());

    await expect.poll(() => onFocusOutside.mock.calls.length).toBe(1);
    await act(nextFrame);
    expect(dialog.getAttribute('data-state')).toBe('open');
  });
});
