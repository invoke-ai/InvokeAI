import { ChakraProvider, Menu, Popover, Portal } from '@chakra-ui/react';
import { system } from '@theme/system';
import { Activity, act, createElement, useCallback, useState, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { MenuContent } from './Menu';
import { PopoverContent } from './Popover';
import { closeWidgetOverlays, WidgetOverlayOwnerContext } from './widgetOverlays';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** Lets the overlay library's deferred close and exit animation land, polling rather than sleeping. */
const settle = (done: () => boolean): Promise<void> =>
  act(
    () =>
      new Promise<void>((resolve, reject) => {
        const deadline = Date.now() + 2000;
        const tick = () => {
          if (done()) {
            resolve();
          } else if (Date.now() > deadline) {
            reject(new Error('overlay never settled'));
          } else {
            globalThis.setTimeout(tick, 20);
          }
        };
        tick();
      })
  );

type Kind = 'menu' | 'popover';

const Overlay = ({ kind, onOpenChange }: { kind: Kind; onOpenChange: (open: boolean) => void }) => {
  const [open, setOpen] = useState(true);
  const handleOpenChange = useCallback(
    ({ open: next }: { open: boolean }) => {
      setOpen(next);
      onOpenChange(next);
    },
    [onOpenChange]
  );
  return kind === 'menu' ? (
    <Menu.Root lazyMount open={open} unmountOnExit onOpenChange={handleOpenChange}>
      <Menu.Trigger>Actions</Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent>
            <Menu.Item value="one">One</Menu.Item>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  ) : (
    <Popover.Root lazyMount open={open} unmountOnExit onOpenChange={handleOpenChange}>
      <Popover.Trigger>Settings</Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <PopoverContent>Panel</PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

/** A widget host: the widget's subtree is hidden when it is switched away, like the shell's Activity boundary. */
const Host = ({ children }: { children: ReactNode }) => {
  const [visible, setVisible] = useState(true);
  // The store closes widget overlays before the switch that hides the widget.
  const switchAway = () => {
    closeWidgetOverlays();
    setVisible(false);
  };
  const switchBack = () => setVisible(true);
  return (
    <ChakraProvider value={system}>
      <button data-role="away" type="button" onClick={switchAway}>
        switch away
      </button>
      <button data-role="back" type="button" onClick={switchBack}>
        switch back
      </button>
      <Activity mode={visible ? 'visible' : 'hidden'}>
        <WidgetOverlayOwnerContext value>{children}</WidgetOverlayOwnerContext>
      </Activity>
    </ChakraProvider>
  );
};

/** An overlay the user can still see; exit-animating closed content does not count. */
const openLayer = (scope: Kind): Element | null =>
  document.querySelector(`[data-scope="${scope}"][data-part="content"][data-state="open"]`);

const render = async (element: ReactNode) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => root!.render(element));
};

const closedLayer = (scope: Kind): boolean =>
  document.querySelector(`[data-scope="${scope}"][data-part="content"]`) === null;

describe.each(['menu', 'popover'] as const)('%s content inside a widget', (kind) => {
  it('leaves the document with the switch that hides the widget, and stays closed when the widget returns', async () => {
    const onOpenChange = vi.fn();
    await render(createElement(Host, null, createElement(Overlay, { kind, onOpenChange })));
    await settle(() => openLayer(kind) !== null);

    await act(() => host!.querySelector<HTMLButtonElement>('[data-role="away"]')!.click());
    expect(openLayer(kind)).toBeNull();

    await act(() => host!.querySelector<HTMLButtonElement>('[data-role="back"]')!.click());
    await settle(() => closedLayer(kind) && onOpenChange.mock.lastCall?.[0] === false);
    expect(openLayer(kind)).toBeNull();

    // The overlay is fully usable again: a fresh opening is a new epoch.
    await act(() => host!.querySelector<HTMLButtonElement>(`[data-scope="${kind}"][data-part="trigger"]`)!.click());
    await settle(() => openLayer(kind) !== null);
  });

  it('closes normally when the switch hides another widget', async () => {
    const onOpenChange = vi.fn();
    await render(createElement(Host, null, createElement(Overlay, { kind, onOpenChange })));
    await settle(() => openLayer(kind) !== null);

    await act(() => closeWidgetOverlays());
    expect(openLayer(kind)).toBeNull();
    await settle(() => closedLayer(kind));

    expect(onOpenChange).toHaveBeenLastCalledWith(false);
  });
});
