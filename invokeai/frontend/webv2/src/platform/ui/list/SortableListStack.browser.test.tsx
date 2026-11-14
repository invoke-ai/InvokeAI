/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider, Portal } from '@chakra-ui/react';
import { Dialog } from '@platform/ui/Dialog';
import { system } from '@theme/system';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { SortableListItem, SortableListStack } from './SortableListStack';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { count?: number; name?: string; position?: number }) =>
      options ? `${key}:${options.name}:${options.position}/${options.count}` : key,
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;
const onReorder = vi.fn<(activeId: string, overId: string) => void>();

/** A caller that applies each move, as a store would. */
const Fruit = ({ disabledId }: { disabledId?: string }) => {
  const [items, setItems] = useState(['apple', 'banana', 'cherry']);
  const reorder = (activeId: string, overId: string) => {
    onReorder(activeId, overId);
    setItems((current) => {
      const next = current.filter((id) => id !== activeId);
      next.splice(current.indexOf(overId), 0, activeId);
      return next;
    });
  };

  return (
    <SortableListStack dividers items={items} label="Fruit" onReorder={reorder}>
      {items.map((id) => (
        <SortableListItem
          key={id}
          actions={<button type="button">Edit {id}</button>}
          handleLabel={`Reorder ${id}`}
          id={id}
          isDisabled={id === disabledId}
          title={id}
        />
      ))}
    </SortableListStack>
  );
};

beforeEach(() => {
  onReorder.mockClear();
  host = document.createElement('div');
  host.style.cssText = 'width:320px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const render = async (disabledId?: string) => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <Fruit disabledId={disabledId} />
      </ChakraProvider>
    )
  );
};

const renderInDialog = async (onOpenChange: (details: { open: boolean }) => void) => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <Dialog.Root open onOpenChange={onOpenChange}>
          <Portal>
            <Dialog.Positioner>
              <Dialog.Content>
                <Fruit />
              </Dialog.Content>
            </Dialog.Positioner>
          </Portal>
        </Dialog.Root>
      </ChakraProvider>
    )
  );
  // The dialog places its initial focus after mount; focusing a handle before that lands would be undone by it.
  await expect.poll(() => document.querySelector('[role="dialog"]')?.contains(document.activeElement)).toBe(true);
};

const order = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[data-list-row]')).map((row) => row.dataset.listRow);
const row = (id: string) => document.querySelector<HTMLElement>(`[data-list-row="${id}"]`)!;
const handle = (id: string) => document.querySelector<HTMLButtonElement>(`button[aria-label="Reorder ${id}"]`)!;
const announcement = () => document.querySelector('[role="status"][aria-live]')?.textContent ?? '';
const settle = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 30);
      })
  );

const pointer = (type: string, target: EventTarget, x: number, y: number) =>
  target.dispatchEvent(new PointerEvent(type, { bubbles: true, button: 0, clientX: x, clientY: y, isPrimary: true }));

describe('SortableListStack', () => {
  it('renders rows as list items whose handle nests in no other control', async () => {
    await render();

    const list = document.querySelector('[role="list"]')!;
    expect(list.getAttribute('aria-label')).toBe('Fruit');
    expect(Array.from(list.querySelectorAll(':scope > [role="listitem"]'), (item) => item.textContent)).toEqual([
      'appleEdit apple',
      'bananaEdit banana',
      'cherryEdit cherry',
    ]);
    expect(handle('apple').parentElement!.closest('button')).toBeNull();
    expect(handle('apple').closest('[data-list-surface]')).toBe(row('apple').querySelector('[data-list-surface]'));
    expect(document.querySelectorAll('[data-list-divider]')).toHaveLength(2);
  });

  it('moves a row dragged by its handle and marks the drag while it lasts', async () => {
    await render();
    const from = handle('apple').getBoundingClientRect();
    const to = row('cherry').getBoundingClientRect();
    const x = from.left + from.width / 2;
    const startY = from.top + from.height / 2;
    const endY = to.top + to.height / 2;

    // Stepped moves, as a hand makes them: the sorting strategy measures rows between moves.
    pointer('pointerdown', handle('apple'), x, startY);
    for (const progress of [0.1, 0.4, 0.7, 1, 1]) {
      pointer('pointermove', document, x, startY + (endY - startY) * progress);
      await settle();
    }

    expect(row('apple').hasAttribute('data-dragging')).toBe(true);
    expect(document.querySelector('[role="list"]')!.hasAttribute('data-sorting')).toBe(true);

    pointer('pointerup', document, x, endY);
    await settle();

    expect(onReorder).toHaveBeenCalledExactlyOnceWith('apple', 'cherry');
    expect(order()).toEqual(['banana', 'cherry', 'apple']);
    expect(row('apple').hasAttribute('data-dragging')).toBe(false);
    expect(document.querySelector('[role="list"]')!.hasAttribute('data-sorting')).toBe(false);
  });

  it('reorders from the keyboard and announces each step by title and position', async () => {
    await render();
    handle('banana').focus();

    await act(() => userEvent.keyboard('{Enter}'));
    await settle();
    expect(handle('banana').getAttribute('aria-pressed')).toBe('true');
    expect(announcement()).toBe('common.list.reorderPickedUp:banana:2/3');

    await act(() => userEvent.keyboard('{ArrowDown}'));
    await settle();
    expect(announcement()).toBe('common.list.reorderMoved:banana:3/3');

    await act(() => userEvent.keyboard('{Enter}'));
    await settle();
    expect(onReorder).toHaveBeenCalledExactlyOnceWith('banana', 'cherry');
    expect(order()).toEqual(['apple', 'cherry', 'banana']);
    expect(announcement()).toBe('common.list.reorderDropped:banana:3/3');
    expect(document.activeElement).toBe(handle('banana'));
  });

  it('puts a keyboard-lifted row back on Escape', async () => {
    await render();
    handle('apple').focus();

    await act(() => userEvent.keyboard('{Enter}'));
    await settle();
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await settle();
    await act(() => userEvent.keyboard('{Escape}'));
    await settle();

    expect(onReorder).not.toHaveBeenCalled();
    expect(order()).toEqual(['apple', 'banana', 'cherry']);
    expect(announcement()).toBe('common.list.reorderCancelled:apple:1/3');
  });

  it('spends Escape on the sort, not on a dialog around the list', async () => {
    const onOpenChange = vi.fn();
    await renderInDialog(onOpenChange);
    handle('apple').focus();

    await act(() => userEvent.keyboard('{Enter}'));
    await settle();
    await act(() => userEvent.keyboard('{Escape}'));
    await settle();

    expect(announcement()).toBe('common.list.reorderCancelled:apple:1/3');
    expect(onOpenChange).not.toHaveBeenCalled();

    await act(() => userEvent.keyboard('{Escape}'));
    expect(onOpenChange).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ open: false }));
  });

  it('does not lift a disabled row', async () => {
    await render('apple');
    const from = handle('apple').getBoundingClientRect();
    const to = row('cherry').getBoundingClientRect();
    const x = from.left + from.width / 2;

    expect(handle('apple').disabled).toBe(true);
    pointer('pointerdown', handle('apple'), x, from.top + from.height / 2);
    for (const y of [to.top, to.top + to.height / 2]) {
      pointer('pointermove', document, x, y);
      await settle();
    }
    pointer('pointerup', document, x, to.top + to.height / 2);
    await settle();

    expect(onReorder).not.toHaveBeenCalled();
    expect(order()).toEqual(['apple', 'banana', 'cherry']);
  });
});
