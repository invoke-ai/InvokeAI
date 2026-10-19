/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { createExternalStore } from '@platform/state/externalStore';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Dialog } from '@platform/ui/Dialog';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFloatingWindowFocus } from '@workbench/focusRegions';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, StrictMode, useCallback } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

/**
 * The real hotkey runtime beneath the real shared dialogs: while a modal dialog is open, application shortcuts
 * behind it do not run, and they return the moment it closes. Only the workbench context plumbing and the
 * first-party command registrations (which need the whole application) are replaced.
 */

const runtimeMocks = vi.hoisted(() => ({
  extensions: null as unknown as ExtensionRegistry,
  store: null as unknown as WorkbenchInternalStore,
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected) =>
    selector(runtimeMocks.store.getSnapshot().activeProject),
  useOptionalWorkbenchExtensions: () => runtimeMocks.extensions,
  useWorkbenchExtensions: () => runtimeMocks.extensions,
  useWorkbenchInternalStore: () => runtimeMocks.store,
  useWorkbenchQueries: () => runtimeMocks.store.queries,
  useWorkbenchSubscription: () => runtimeMocks.store.subscribe,
}));
vi.mock('./firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));

import { WorkbenchHotkeyRuntime } from './WorkbenchHotkeyRuntime';

const scene = createExternalStore({
  confirm: false,
  confirmHost: true,
  nonModal: false,
  outer: false,
  rename: false,
});
const setScene = (patch: Partial<ReturnType<typeof scene.getSnapshot>>) => act(() => scene.patchSnapshot(patch));

let ran: string[] = [];
let confirmed = 0;
let confirmExited = 0;
let renamedTo: string[] = [];

const Scene = () => {
  const { confirm, confirmHost, nonModal, outer, rename } = scene.useSnapshot();
  const { activate } = useFloatingWindowFocus('image-map', runtimeMocks.store.getSnapshot().activeProject.id);
  const handlePointerDown = useCallback(() => activate({ byPointer: true }), [activate]);

  return (
    <>
      <div
        data-floating-window="image-map"
        data-hotkey-widget-instance-id="image-map"
        data-hotkey-widget-region="floating"
        data-hotkey-widget-type-id="image-map"
        data-testid="window"
        onPointerDownCapture={handlePointerDown}
      >
        <button type="button">Window control</button>
      </div>
      {confirmHost ? (
        <ConfirmDialog
          body="Delete the selected image?"
          confirmLabel="Delete"
          isOpen={confirm}
          title="Delete image"
          onClose={() => scene.patchSnapshot({ confirm: false })}
          onConfirm={() => {
            confirmed += 1;
          }}
          onExitComplete={() => {
            confirmExited += 1;
          }}
        />
      ) : null}
      <RenameDialog
        initialName="Sunset"
        isOpen={rename}
        onClose={() => scene.patchSnapshot({ rename: false })}
        onSubmit={(name) => {
          renamedTo.push(name);
        }}
      />
      <Dialog.Root open={outer} onOpenChange={({ open }) => scene.patchSnapshot({ outer: open })}>
        <Dialog.Positioner>
          <Dialog.Content aria-label="Outer">
            <Dialog.Body>
              <Button onClick={() => scene.patchSnapshot({ confirm: true })}>Open inner</Button>
            </Dialog.Body>
          </Dialog.Content>
        </Dialog.Positioner>
      </Dialog.Root>
      <Dialog.Root modal={false} open={nonModal}>
        <Dialog.Positioner>
          <Dialog.Content aria-label="Non-modal">
            <Dialog.Body>Details</Dialog.Body>
          </Dialog.Content>
        </Dialog.Positioner>
      </Dialog.Root>
    </>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** `mod` is Control off macOS, where these browser tests run. */
const invoke = () => userEvent.keyboard('{Control>}{Enter}{/Control}');
const runWidgetCommand = () => userEvent.keyboard('j');
const confirmDialog = () => page.getByRole('alertdialog', { name: 'Delete image' });
const renameDialog = () => page.getByRole('dialog', { name: 'Rename project' });
/** Waits out the opening animation, so a close that follows animates out rather than cutting it short. */
const openConfirm = async () => {
  await setScene({ confirm: true });
  await expect.element(confirmDialog()).toBeVisible();
  await expect.poll(() => confirmDialog().element().getAnimations().length).toBe(0);
};
/** Dispatched in the same task as the state change that precedes it, so no animation frame can intervene. */
const dispatchKey = (init: KeyboardEventInit) =>
  (document.activeElement ?? document.body).dispatchEvent(
    new KeyboardEvent('keydown', { bubbles: true, cancelable: true, code: init.key, ...init })
  );

beforeEach(async () => {
  ran = [];
  confirmed = 0;
  confirmExited = 0;
  renamedTo = [];
  scene.setSnapshot({ confirm: false, confirmHost: true, nonModal: false, outer: false, rename: false });
  runtimeMocks.store = createWorkbenchStore();
  runtimeMocks.store.commands.widgets.float('image-map');
  runtimeMocks.extensions = createExtensionRegistry();
  const { commands, hotkeys } = runtimeMocks.extensions;
  const source = {
    instanceId: 'image-map',
    projectId: runtimeMocks.store.getSnapshot().activeProject.id,
    region: 'floating' as const,
    typeId: 'image-map',
  };

  // The first-party catalog binds Mod+Enter to invoke and Mod+K to the palette; the handlers stand in for the app's.
  commands.register({ handler: () => ran.push('app.invoke'), id: 'app.invoke', title: 'Invoke' });
  commands.register({ handler: () => ran.push('app.openCommandPalette'), id: 'app.openCommandPalette', title: 'P' });
  // Widget shortcuts on a plain key and on the keys a dialog closes with, as Gallery and Canvas bind them.
  for (const [id, key] of [
    ['map.widget', 'j'],
    ['map.escape', 'esc'],
    ['map.enter', 'enter'],
  ] as const) {
    commands.register({ handler: () => ran.push(id), id, source, title: id });
    hotkeys.register({ commandId: id, defaultKeys: [key], id, scope: 'widget', source, title: id });
  }

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <StrictMode>
        <ChakraProvider value={system}>
          <WorkbenchFocusProvider>
            <WorkbenchHotkeyRuntime />
            <Scene />
          </WorkbenchFocusProvider>
        </ChakraProvider>
      </StrictMode>
    )
  );
  // The widget command needs a target: the floating window holds focus until a dialog takes it.
  await act(() =>
    host!.querySelector('[data-testid="window"]')!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
  );
  await runWidgetCommand();
  await invoke();
  expect(ran).toEqual(['map.widget', 'app.invoke']);
  ran = [];
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('WorkbenchHotkeyRuntime under a modal dialog', () => {
  it('runs nothing behind a confirmation, which keeps its own keys and gives shortcuts back once it closes', async () => {
    await openConfirm();

    await invoke();
    await runWidgetCommand();
    expect(ran).toEqual([]);
    await expect.element(confirmDialog()).toBeVisible();

    await userEvent.keyboard('{Escape}');
    await expect.element(confirmDialog()).not.toBeInTheDocument();
    expect(confirmed).toBe(0);

    await openConfirm();
    page.getByRole('button', { name: 'Delete' }).element().focus();
    await userEvent.keyboard('{Enter}');
    await expect.element(confirmDialog()).not.toBeInTheDocument();
    expect(confirmed).toBe(1);

    await invoke();
    await runWidgetCommand();
    expect(ran).toEqual(['app.invoke', 'map.widget']);
  });

  it('runs nothing behind for the Escape or Enter that closes a dialog, and the next press reaches it', async () => {
    await openConfirm();
    await userEvent.keyboard('{Escape}');
    expect(ran).toEqual([]);
    expect(confirmDialog().element().getAttribute('data-state')).toBe('closed');
    dispatchKey({ key: 'Escape' });
    expect(ran).toEqual(['map.escape']);

    await openConfirm();
    page.getByRole('button', { name: 'Delete' }).element().focus();
    await userEvent.keyboard('{Enter}');
    expect(confirmed).toBe(1);
    expect(ran).toEqual(['map.escape']);
    dispatchKey({ key: 'Enter' });
    expect(ran).toEqual(['map.escape', 'map.enter']);
  });

  it('lets a rename be typed and submitted with Enter while shortcuts behind it stay inactive', async () => {
    await setScene({ rename: true });
    const input = page.getByRole('textbox', { name: 'Project name' });
    await expect.element(input).toHaveFocus();

    await userEvent.keyboard('{Control>}a{/Control}Dusk');
    await invoke();
    expect(ran).toEqual([]);
    await expect.element(input).toHaveValue('Dusk');

    await userEvent.keyboard('{Enter}');
    await expect.element(renameDialog()).not.toBeInTheDocument();
    expect(renamedTo).toEqual(['Dusk']);

    await invoke();
    expect(ran).toEqual(['app.invoke']);
  });

  it('gives shortcuts back while the closing dialog animates out, and takes them again if it reopens', async () => {
    await openConfirm();
    const dialog = confirmDialog().element();

    await setScene({ confirm: false });
    // Still on screen, animating out and inert.
    expect(dialog.isConnected).toBe(true);
    expect(dialog.getAttribute('data-state')).toBe('closed');
    dispatchKey({ ctrlKey: true, key: 'Enter' });
    expect(ran).toEqual(['app.invoke']);

    await setScene({ confirm: true });
    expect(confirmExited).toBe(0);
    dispatchKey({ ctrlKey: true, key: 'Enter' });
    expect(ran).toEqual(['app.invoke']);
    await expect.element(confirmDialog()).toBeVisible();
  });

  it('keeps shortcuts inactive while an outer dialog stays open beneath a closed inner one', async () => {
    await setScene({ outer: true });
    await page.getByRole('button', { name: 'Open inner' }).click();
    await expect.element(confirmDialog()).toBeVisible();

    await page.getByRole('button', { name: 'Cancel' }).click();
    await expect.element(confirmDialog()).not.toBeInTheDocument();
    await expect.element(page.getByRole('dialog', { name: 'Outer' })).toBeVisible();
    await invoke();
    expect(ran).toEqual([]);

    await userEvent.keyboard('{Escape}');
    await expect.element(page.getByRole('dialog', { name: 'Outer' })).not.toBeInTheDocument();
    await invoke();
    expect(ran).toEqual(['app.invoke']);
  });

  it('gives shortcuts back when an open dialog is unmounted without closing', async () => {
    await openConfirm();

    await setScene({ confirmHost: false });
    await expect.element(confirmDialog()).not.toBeInTheDocument();
    await invoke();
    expect(ran).toEqual(['app.invoke']);
  });

  it('still opens the command palette above a dialog', async () => {
    await openConfirm();

    await userEvent.keyboard('{Control>}k{/Control}');
    expect(ran).toEqual(['app.openCommandPalette']);
  });

  it('leaves shortcuts active beside a non-modal dialog', async () => {
    await setScene({ nonModal: true });
    await expect.element(page.getByRole('dialog', { name: 'Non-modal' })).toBeVisible();

    await act(() => (document.activeElement as HTMLElement | null)?.blur());
    dispatchKey({ ctrlKey: true, key: 'Enter' });
    expect(ran).toEqual(['app.invoke']);
  });

  it('runs nothing for a key pressed during IME composition', () => {
    dispatchKey({ ctrlKey: true, isComposing: true, key: 'Enter' });
    dispatchKey({ ctrlKey: true, key: 'Enter', keyCode: 229 } as KeyboardEventInit);
    expect(ran).toEqual([]);

    dispatchKey({ ctrlKey: true, key: 'Enter' });
    expect(ran).toEqual(['app.invoke']);
  });
});
