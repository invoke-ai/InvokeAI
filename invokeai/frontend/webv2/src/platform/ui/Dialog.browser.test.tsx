import { ChakraProvider, Menu, Portal } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, StrictMode, useCallback, useState, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { Dialog } from './Dialog';
import { MenuContent, MenuItem } from './Menu';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** As the app mounts it: under StrictMode, with a control outside every dialog. */
const render = async (content: ReactNode) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <StrictMode>
        <ChakraProvider value={system}>
          <button type="button">Outside</button>
          {content}
        </ChakraProvider>
      </StrictMode>
    )
  );
};

const dialogNamed = (title: string) => page.getByRole('dialog', { name: title });
const isOpen = (title: string) => dialogNamed(title).query()?.getAttribute('data-state') === 'open';

const SimpleDialog = ({
  children,
  modal,
  onFocusOutside,
  onOpenChange,
  open,
  title,
}: {
  children?: ReactNode;
  modal?: boolean;
  onFocusOutside?: () => void;
  onOpenChange: (details: Dialog.OpenChangeDetails) => void;
  open: boolean;
  title: string;
}) => (
  <Dialog.Root
    closeOnInteractOutside
    modal={modal}
    open={open}
    onFocusOutside={onFocusOutside}
    onOpenChange={onOpenChange}
  >
    <Portal>
      {modal === false ? null : <Dialog.Backdrop />}
      <Dialog.Positioner>
        <Dialog.Content>
          <Dialog.Header>
            <Dialog.Title>{title}</Dialog.Title>
          </Dialog.Header>
          <Dialog.Body>{children}</Dialog.Body>
        </Dialog.Content>
      </Dialog.Positioner>
    </Portal>
  </Dialog.Root>
);

const getEditInitialFocus = () => document.querySelector<HTMLInputElement>('input[name="item-name"]');

/**
 * A dialog that opens a second one over itself, mounted open and released after it animates out, as the layout preset
 * manager opens its edit dialog.
 */
const StackedDialogs = () => {
  const [isManagerOpen, setIsManagerOpen] = useState(true);
  const [isEditMounted, setIsEditMounted] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const openEdit = () => {
    setIsEditMounted(true);
    setIsEditOpen(true);
  };
  const changeManagerOpen = useCallback((details: Dialog.OpenChangeDetails) => setIsManagerOpen(details.open), []);
  const changeEditOpen = useCallback((details: Dialog.OpenChangeDetails) => setIsEditOpen(details.open), []);
  const releaseEdit = useCallback(() => setIsEditMounted(false), []);

  return (
    <>
      <SimpleDialog open={isManagerOpen} title="Manager" onOpenChange={changeManagerOpen}>
        <button type="button" onClick={openEdit}>
          Edit
        </button>
      </SimpleDialog>
      {isEditMounted ? (
        <Dialog.Root
          initialFocusEl={getEditInitialFocus}
          lazyMount
          open={isEditOpen}
          unmountOnExit
          onExitComplete={releaseEdit}
          onOpenChange={changeEditOpen}
        >
          <Portal>
            <Dialog.Backdrop />
            <Dialog.Positioner>
              <Dialog.Content>
                <Dialog.Header>
                  <Dialog.Title>Edit item</Dialog.Title>
                </Dialog.Header>
                <Dialog.Body>
                  <input aria-label="Name" name="item-name" />
                </Dialog.Body>
                <Dialog.Footer>
                  <button type="button" onClick={() => setIsEditOpen(false)}>
                    Cancel
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

/** A modal or non-modal dialog the test opens and observes. */
const Observed = ({
  children,
  modal,
  onFocusOutside,
}: {
  children?: ReactNode;
  modal?: boolean;
  onFocusOutside?: () => void;
}) => {
  const [open, setOpen] = useState(true);
  const changeOpen = useCallback((details: Dialog.OpenChangeDetails) => setOpen(details.open), []);
  return (
    <SimpleDialog modal={modal} open={open} title="Observed" onFocusOutside={onFocusOutside} onOpenChange={changeOpen}>
      {children ?? <button type="button">Inside</button>}
    </SimpleDialog>
  );
};

/** The dialog's Escape and outside handling is armed in the frame after it opens. */
const settled = async (title: string) => {
  await expect.poll(() => isOpen(title)).toBe(true);
  await act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      })
  );
};

describe('Dialog', () => {
  it.each([
    ['Cancel', () => userEvent.click(page.getByRole('button', { name: 'Cancel' }))],
    ['Escape', () => userEvent.keyboard('{Escape}')],
  ])('closes only a dialog stacked over another on %s, leaving focus in the one below', async (_way, dismiss) => {
    await render(<StackedDialogs />);
    await settled('Manager');

    for (let attempt = 0; attempt < 2; attempt += 1) {
      await act(() => userEvent.click(page.getByRole('button', { name: 'Edit' })));
      await settled('Edit item');
      await expect.poll(() => dialogNamed('Edit item').element().contains(document.activeElement)).toBe(true);

      await act(dismiss);

      await expect.poll(() => dialogNamed('Edit item').query()).toBeNull();
      expect(isOpen('Manager')).toBe(true);
      await expect.poll(() => document.activeElement).toBe(page.getByRole('button', { name: 'Edit' }).element());
    }
  });

  it('keeps a modal open when focus is moved outside it, still telling its own handler', async () => {
    const onFocusOutside = vi.fn();
    await render(<Observed onFocusOutside={onFocusOutside} />);
    await settled('Observed');

    await act(() => host!.querySelector<HTMLButtonElement>('button')!.focus());

    await expect.poll(() => onFocusOutside).toHaveBeenCalled();
    // Two frames on, so a dismissal the handler let through has re-rendered the dialog closed.
    await settled('Observed');
    expect(isOpen('Observed')).toBe(true);
  });

  it.each([
    [
      'a press outside',
      () => userEvent.click(document.querySelector('[data-part="positioner"]')!, { position: { x: 4, y: 4 } }),
    ],
    ['Escape', () => userEvent.keyboard('{Escape}')],
  ])('dismisses a modal on %s', async (_way, dismiss) => {
    await render(<Observed />);
    await settled('Observed');

    await act(dismiss);

    await expect.poll(() => isOpen('Observed')).toBe(false);
  });

  it('dismisses a non-modal dialog when focus moves outside it, as before', async () => {
    await render(<Observed modal={false} />);
    await settled('Observed');

    await act(() => host!.querySelector<HTMLButtonElement>('button')!.focus());

    await expect.poll(() => isOpen('Observed')).toBe(false);
  });

  it('stays open while a menu opened inside it is used', async () => {
    const onSelect = vi.fn();
    await render(
      <Observed>
        <Menu.Root>
          <Menu.Trigger>Actions</Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent>
                <MenuItem value="rename" onSelect={onSelect}>
                  Rename
                </MenuItem>
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Observed>
    );
    await settled('Observed');

    await userEvent.click(page.getByRole('button', { name: 'Actions' }));
    // The menu's content is portalled outside the dialog's, yet nested in its layer.
    await expect.poll(() => document.querySelector('[role="menu"]')?.contains(document.activeElement)).toBe(true);
    await userEvent.click(page.getByRole('menuitem', { name: 'Rename' }));

    expect(onSelect).toHaveBeenCalledOnce();
    await expect.poll(() => document.querySelector('[role="menu"]')?.checkVisibility() ?? false).toBe(false);
    expect(isOpen('Observed')).toBe(true);
    await expect.element(page.getByRole('button', { name: 'Actions' })).toHaveFocus();
  });
});
