import { ChakraProvider } from '@chakra-ui/react';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { createWorkbenchStore, type WorkbenchInternalStore } from '@workbench/workbenchStore';
import { act, StrictMode, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

let store: WorkbenchInternalStore;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { name?: string }) =>
      ({
        'common.done': 'Done',
        'topbar.presets.edit': 'Edit layout preset',
        'topbar.presets.deleteNamed': `Delete ${options?.name ?? ''}`,
        'topbar.presets.editNamed': `Edit ${options?.name ?? ''}`,
        'topbar.presets.manage': 'Manage presets',
        'topbar.presets.reorderNamed': `Reorder ${options?.name ?? ''}`,
        'topbar.presets.restore': 'Restore shipped preset defaults',
        'topbar.presets.restoreNamed': `Restore ${options?.name ?? ''} defaults`,
      })[key] ?? key,
  }),
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => store.commands,
  useWorkbenchSelector: <Selected,>(
    selector: (snapshot: ReturnType<WorkbenchInternalStore['getSnapshot']>) => Selected
  ) => selector(useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)),
}));

import { LayoutPresetAdminDialogs } from './LayoutPresetAdminDialogs';
import { LayoutPresetManagerDialog } from './LayoutPresetManagerDialog';
import { LayoutPresetManagerDialogBody } from './LayoutPresetManagerDialogBody';
import { closeLayoutPresetManager, openLayoutPresetManager } from './layoutPresetManagerStore';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const noop = () => {};

const renderManager = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <LayoutPresetManagerDialogBody isOpen onExitComplete={noop} />
      </ChakraProvider>
    );
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });
};

const presetRowIds = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[data-list-row]')).map((element) => element.dataset.listRow);

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

const dragHandle = (name: string): HTMLButtonElement | null =>
  document.querySelector<HTMLButtonElement>(`button[aria-label="Reorder ${name}"]`);

const presetRow = (id: string): HTMLElement | null => document.querySelector<HTMLElement>(`[data-list-row="${id}"]`);

/** Both topbar hosts, as the app mounts them (StrictMode included), so the edit dialog stacks over the manager. */
const renderTopBarDialogs = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <StrictMode>
        <ChakraProvider value={system}>
          <LayoutPresetManagerDialog />
          <LayoutPresetAdminDialogs />
        </ChakraProvider>
      </StrictMode>
    )
  );
};

const dialogNamed = (title: string): HTMLElement | null =>
  Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"]')).find(
    (dialog) => dialog.querySelector('[data-part="title"]')?.textContent === title
  ) ?? null;

const buttonNamed = (container: Element, text: string): HTMLButtonElement =>
  Array.from(container.querySelectorAll('button')).find((button) => button.textContent === text)!;

const editButton = (name: string): HTMLButtonElement | null =>
  document.querySelector<HTMLButtonElement>(`button[aria-label="Edit ${name}"]`);

beforeEach(() => {
  store = createWorkbenchStore();
  store.commands.layout.createPreset('custom-1', 'Custom', 'star');
  store.commands.layout.reorderPresets('custom-1', 'edit');
  store.commands.layout.renamePreset('compose', 'Writing');
});

afterEach(async () => {
  closeLayoutPresetManager();
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('LayoutPresetManagerDialogBody', () => {
  it('animates the lazily loaded manager out instead of unmounting it on close', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <LayoutPresetManagerDialog />
        </ChakraProvider>
      )
    );
    await act(() => openLayoutPresetManager());
    await expect.poll(() => document.querySelector('[role="dialog"]')?.getAttribute('data-state')).toBe('open');
    const dialog = document.querySelector('[role="dialog"]')!;
    const done = Array.from(dialog.querySelectorAll('button')).find((button) => button.textContent === 'Done')!;

    const frames = await recordDialogExit(dialog, () => act(() => userEvent.click(done)));

    // An unmounted dialog never reaches its closed state, so it cannot animate out; a retained one does, then leaves.
    expect(closingFrames(frames)).not.toHaveLength(0);
    await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();
  });

  it.each([
    ['Cancel', () => userEvent.click(buttonNamed(dialogNamed('Edit layout preset')!, 'common.cancel'))],
    ['Escape', () => userEvent.keyboard('{Escape}')],
    [
      'the close button',
      () => userEvent.click(dialogNamed('Edit layout preset')!.querySelector('[data-part="close-trigger"]')!),
    ],
  ])('closes only the edit dialog stacked over the manager on %s', async (_way, dismiss) => {
    await renderTopBarDialogs();
    await act(() => openLayoutPresetManager());
    await expect.poll(() => dialogNamed('Manage presets')?.getAttribute('data-state')).toBe('open');

    for (let attempt = 0; attempt < 2; attempt += 1) {
      await act(() => userEvent.click(editButton('Writing')!));
      await expect.poll(() => dialogNamed('Edit layout preset')?.getAttribute('data-state')).toBe('open');
      await expect.poll(() => dialogNamed('Edit layout preset')?.contains(document.activeElement)).toBe(true);

      await act(dismiss);

      await expect.poll(() => dialogNamed('Edit layout preset')).toBeNull();
      expect(dialogNamed('Manage presets')?.getAttribute('data-state')).toBe('open');
      await expect.poll(() => document.activeElement).toBe(editButton('Writing'));
    }
  });

  it('renders one account-ordered list without preset descriptions', async () => {
    await renderManager();

    expect(presetRowIds()).toEqual(['compose', 'custom-1', 'edit', 'video', 'automate']);
    expect(document.body.textContent).toContain('Writing');
    expect(document.body.textContent).not.toContain('Text to image');
    expect(document.body.textContent).not.toContain('Canvas editing');
  });

  it('reorders presets by dragging a row handle', async () => {
    await renderManager();
    const handle = dragHandle('Writing');
    const target = presetRow('edit');
    expect(handle).not.toBeNull();
    expect(target).not.toBeNull();
    const start = handle!.getBoundingClientRect();
    const end = target!.getBoundingClientRect();
    const startX = start.left + start.width / 2;
    const startY = start.top + start.height / 2;
    const endY = end.top + end.height / 2;

    await act(() => pointer('pointerdown', handle!, startX, startY));
    await act(() => pointer('pointermove', handle!.ownerDocument, startX, startY + 8));
    await act(() => pointer('pointermove', handle!.ownerDocument, startX, endY));
    await act(() => pointer('pointerup', handle!.ownerDocument, startX, endY));

    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['custom-1', 'edit', 'compose', 'video', 'automate']);
  });

  it('reorders presets from a focused row handle with the keyboard', async () => {
    await renderManager();
    // The dialog places its initial focus asynchronously after mount; focusing
    // the handle before that lands would be undone by it.
    await expect.poll(() => document.querySelector('[role="dialog"]')?.contains(document.activeElement)).toBe(true);
    const handle = dragHandle('Custom');
    expect(handle).not.toBeNull();
    handle!.focus();

    await act(() => userEvent.keyboard('{Enter}'));
    await act(
      () =>
        new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 0);
        })
    );
    expect(handle).toHaveAttribute('aria-pressed', 'true');
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await act(
      () =>
        new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 0);
        })
    );
    await act(() => userEvent.keyboard('{Enter}'));

    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['compose', 'edit', 'custom-1', 'video', 'automate']);
  });
});
