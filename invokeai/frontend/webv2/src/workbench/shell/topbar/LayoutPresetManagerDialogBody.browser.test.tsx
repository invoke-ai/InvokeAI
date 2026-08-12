import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createWorkbenchStore, type WorkbenchInternalStore } from '@workbench/workbenchStore';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

let store: WorkbenchInternalStore;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { name?: string }) =>
      ({
        'common.done': 'Done',
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

import { LayoutPresetManagerDialogBody } from './LayoutPresetManagerDialogBody';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderManager = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <LayoutPresetManagerDialogBody />
      </ChakraProvider>
    );
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });
};

const presetRowIds = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[data-layout-preset-id]')).map(
    (element) => element.dataset.layoutPresetId
  );

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

const dragHandle = (name: string): HTMLButtonElement | null =>
  document.querySelector<HTMLButtonElement>(`button[aria-label="Reorder ${name}"]`);

const presetRow = (id: string): HTMLElement | null =>
  document.querySelector<HTMLElement>(`[data-layout-preset-id="${id}"]`);

beforeEach(() => {
  store = createWorkbenchStore();
  store.commands.layout.createPreset('custom-1', 'Custom', 'star');
  store.commands.layout.reorderPresets('custom-1', 'edit');
  store.commands.layout.renamePreset('compose', 'Writing');
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('LayoutPresetManagerDialogBody', () => {
  it('renders one account-ordered list without preset descriptions', async () => {
    await renderManager();

    expect(presetRowIds()).toEqual(['compose', 'custom-1', 'edit', 'automate']);
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

    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['custom-1', 'edit', 'compose', 'automate']);
  });

  it('reorders presets from a focused row handle with the keyboard', async () => {
    await renderManager();
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

    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['compose', 'edit', 'custom-1', 'automate']);
  });
});
