import { ChakraProvider, Input, Menu, Portal } from '@chakra-ui/react';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ListRowProps } from './list/List';

import { Button } from './Button';
import { ConfirmDialog } from './ConfirmDialog';
import { List } from './list/List';
import { ListItem } from './list/ListItem';
import { listRowsFromItems } from './list/listRows';
import { ManagerColumn, ManagerDetailHeader, ManagerLayout } from './ManagerLayout';
import { MenuContent } from './Menu';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** Both panes need 50rem (800px). */
const WIDE_PX = 900;
const NARROW_PX = 640;

const ITEMS = Array.from({ length: 60 }, (_, index) => ({ id: `item-${index}`, name: `Item ${index}` }));

/**
 * A manager in miniature: rows open the detail, a portaled menu can too, the detail deletes its item through a
 * confirmation, and a footer reports progress under both panes.
 */
const Harness = () => {
  const [items, setItems] = useState(ITEMS);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isQueueMaximized, setIsQueueMaximized] = useState(false);
  const open = (id: string) => {
    setActiveId(id);
    setIsDetailOpen(true);
  };

  return (
    <ManagerLayout
      backLabel="Back to items"
      detail={
        <>
          <ManagerDetailHeader>
            <h3>{activeId ?? 'Nothing selected'}</h3>
          </ManagerDetailHeader>
          <Input aria-label="Draft" />
          <button type="button">Primary action</button>
          <Button onClick={() => setIsDeleteOpen(true)}>Delete item</Button>
          <ConfirmDialog
            body="Gone for good."
            confirmLabel="Delete for good"
            isOpen={isDeleteOpen}
            title="Delete item?"
            onClose={() => setIsDeleteOpen(false)}
            onConfirm={() => {
              setItems((current) => current.filter((item) => item.id !== activeId));
              setActiveId(null);
              setIsDetailOpen(false);
            }}
          />
        </>
      }
      footer={
        <div>
          <button type="button">Footer progress</button>
          <button type="button" onClick={() => setIsQueueMaximized((current) => !current)}>
            {isQueueMaximized ? 'Restore queue' : 'Maximize queue'}
          </button>
        </div>
      }
      footerFills={isQueueMaximized}
      isDetailOpen={isDetailOpen}
      library={
        <ManagerColumn
          actions={
            <Menu.Root>
              <Menu.Trigger asChild>
                <Button size="sm">Item actions</Button>
              </Menu.Trigger>
              <Portal>
                <Menu.Positioner>
                  <MenuContent>
                    <Menu.Item value="open" onSelect={() => open('item-5')}>
                      Open item 5
                    </Menu.Item>
                  </MenuContent>
                </Menu.Positioner>
              </Portal>
            </Menu.Root>
          }
          addAction={{ label: 'Add items', onAdd: () => setIsDetailOpen(true) }}
          title="Items"
        >
          <List
            activeKey={activeId}
            label="Item library"
            renderItem={(item: (typeof ITEMS)[number], rowProps: ListRowProps) => (
              <ListItem {...rowProps} title={item.name} onPress={() => open(item.id)} />
            )}
            rows={listRowsFromItems(items, (item) => item.id)}
          />
        </ManagerColumn>
      }
      onBack={() => setIsDetailOpen(false)}
    />
  );
};

let host: HTMLDivElement;
let root: Root;

const nextFrame = () =>
  new Promise((resolve) => {
    requestAnimationFrame(resolve);
  });

const setWidth = async (width: number) => {
  await act(async () => {
    host.style.width = `${String(width)}px`;
    await nextFrame();
    await nextFrame();
  });
};

const render = async (width: number) => {
  host.style.cssText = `height:480px;width:${String(width)}px;`;
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <Harness />
      </ChakraProvider>
    )
  );
};

const library = () => page.getByRole('list', { name: 'Item library' });
const detail = () => page.getByRole('textbox', { name: 'Draft' });
const back = () => page.getByRole('button', { name: 'Back to items' });
const add = () => page.getByRole('button', { name: 'Add items' });
const row = (name: string) => page.getByRole('button', { name, exact: true });
const rowElement = (id: string) => host.querySelector<HTMLElement>(`[data-list-row="${id}"] [data-list-primary]`)!;
const viewport = () => host.querySelector<HTMLElement>('[data-list-viewport]')!;
const draft = () => host.querySelector<HTMLInputElement>('input[aria-label="Draft"]')!;
const focused = () => document.activeElement as HTMLElement;
const isShown = (element: Element) => element.checkVisibility({ visibilityProperty: true });

beforeEach(() => {
  applyThemeToRoot(DEFAULT_THEME_ID);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('ManagerLayout', () => {
  it('shows the library, the detail and the footer side by side, without Back or the Add shortcut', async () => {
    await render(WIDE_PX);

    await expect.element(library()).toBeVisible();
    await expect.element(detail()).toBeVisible();
    await expect.element(page.getByRole('button', { name: 'Footer progress' })).toBeVisible();
    await expect.element(back()).not.toBeInTheDocument();
    await expect.element(add()).not.toBeInTheDocument();
    await row('Item 3').click();
    await expect.element(row('Item 3')).toHaveFocus();
    await expect.element(page.getByRole('heading', { name: 'item-3' })).toBeVisible();
    expect(await auditAccessibility(host)).toEqual([]);
  });

  it('shows one pane at a time when narrow and returns to the row, scroll and focus it came from', async () => {
    await render(NARROW_PX);

    await expect.element(library()).toBeVisible();
    await expect.element(detail()).not.toBeInTheDocument();
    await expect.element(page.getByRole('button', { name: 'Footer progress' })).toBeVisible();
    await act(async () => {
      viewport().scrollTop = 900;
      await nextFrame();
    });
    const origin = rowElement('item-23');
    expect(origin.getBoundingClientRect().top).toBeGreaterThan(host.getBoundingClientRect().top);

    await userEvent.click(origin);

    // Immediate: the panes swap in the same frame, with no transition holding the old one on screen.
    const panes = [...host.querySelectorAll<HTMLElement>('[data-manager-pane]')];
    expect(panes.map((pane) => getComputedStyle(pane).visibility)).toEqual(['hidden', 'visible']);
    expect(panes.flatMap((pane) => pane.getAnimations())).toEqual([]);
    await expect.element(back()).toHaveFocus();
    await expect.element(page.getByRole('heading', { name: 'item-23' })).toBeVisible();
    await expect.element(library()).not.toBeInTheDocument();
    // The footer stays under whichever pane shows.
    await expect.element(page.getByRole('button', { name: 'Footer progress' })).toBeVisible();
    expect(await auditAccessibility(host)).toEqual([]);

    await back().click();

    await expect.element(library()).toBeVisible();
    await expect.element(detail()).not.toBeInTheDocument();
    expect(viewport().scrollTop).toBe(900);
    expect(focused()).toBe(origin);
    expect(origin.getAttribute('aria-current')).toBe('true');
    expect(await auditAccessibility(host)).toEqual([]);
  });

  it('keeps the hidden pane out of the tab order and unfocusable', async () => {
    await render(NARROW_PX);
    await row('Item 1').click();
    await expect.element(back()).toHaveFocus();

    const hiddenRow = rowElement('item-1');
    hiddenRow.focus();
    expect(focused()).not.toBe(hiddenRow);

    const stops: string[] = [];
    for (let index = 0; index < 5; index += 1) {
      await userEvent.tab();
      stops.push(focused().getAttribute('aria-label') ?? focused().textContent);
    }
    expect(stops.some((stop) => stop.startsWith('Item '))).toBe(false);
  });

  it('opens the detail from the Add shortcut, shown only when the panes take turns', async () => {
    await render(NARROW_PX);

    await add().click();

    await expect.element(back()).toHaveFocus();
    await expect.element(detail()).toBeVisible();
    await setWidth(WIDE_PX);
    await expect.element(add()).not.toBeInTheDocument();
  });

  it('keeps the selection, the open detail and its edits across wide and narrow resizes', async () => {
    await render(WIDE_PX);
    await row('Item 4').click();
    await userEvent.fill(draft(), 'unsaved words');

    await setWidth(NARROW_PX);

    await expect.element(detail()).toBeVisible();
    await expect.element(back()).toBeVisible();
    expect(draft().value).toBe('unsaved words');

    await setWidth(WIDE_PX);

    await expect.element(library()).toBeVisible();
    await expect.element(back()).not.toBeInTheDocument();
    expect(draft().value).toBe('unsaved words');
    await expect.element(row('Item 4')).toHaveAttribute('aria-current', 'true');
  });

  it('moves focus off a pane a resize hides, in both directions', async () => {
    await render(WIDE_PX);
    await row('Item 6').click();
    expect(focused()).toBe(rowElement('item-6'));

    // Narrowing with the detail open hides the library holding focus.
    await setWidth(NARROW_PX);
    await vi.waitFor(() => expect(focused()).toBe(back().element()));

    // Widening hides Back itself; focus stays in the detail, on a control that shows.
    await setWidth(WIDE_PX);
    await vi.waitFor(() => {
      expect(isShown(focused())).toBe(true);
      expect(host.querySelector('[data-manager-pane="detail"]')!.contains(focused())).toBe(true);
    });

    // A pane that stays shown keeps its focus.
    await row('Item 7').click();
    await setWidth(NARROW_PX);
    await vi.waitFor(() => expect(focused()).toBe(back().element()));
    await back().click();
    expect(focused()).toBe(rowElement('item-7'));
    await setWidth(WIDE_PX);
    expect(focused()).toBe(rowElement('item-7'));
  });

  it('returns focus to the list after a confirmed delete removes the open row', async () => {
    await render(NARROW_PX);
    await row('Item 3').click();

    await page.getByRole('button', { name: 'Delete item' }).click();
    await page.getByRole('button', { name: 'Delete for good' }).click();

    await expect.element(page.getByRole('alertdialog')).not.toBeInTheDocument();
    await expect.element(library()).toBeVisible();
    await vi.waitFor(() => {
      expect(focused().matches('[data-list-primary]')).toBe(true);
      expect(isShown(focused())).toBe(true);
    });
    expect(rowElement('item-3')).toBeNull();
  });

  it('moves focus into the detail when a portaled menu opens it', async () => {
    await render(NARROW_PX);

    await page.getByRole('button', { name: 'Item actions' }).click();
    await page.getByRole('menuitem', { name: 'Open item 5' }).click();

    await expect.element(page.getByRole('heading', { name: 'item-5' })).toBeVisible();
    await vi.waitFor(() => expect(focused()).toBe(back().element()));
  });
  describe('with the footer filling the panes, as a maximized queue does', () => {
    const grid = () => host.querySelector<HTMLElement>('[data-view]')!.getBoundingClientRect();
    /** Relative to the manager, which focusing a control may scroll within the test frame. */
    const within = (selector: string) => {
      const bounds = host.querySelector<HTMLElement>(selector)!.getBoundingClientRect();
      const origin = grid();

      return {
        bottom: bounds.bottom - origin.top,
        left: bounds.left - origin.left,
        right: bounds.right - origin.left,
        top: bounds.top - origin.top,
      };
    };
    const footer = () => within('[data-manager-footer]');
    const libraryPane = () => within('[data-manager-pane="library"]');
    const toggle = () => page.getByRole('button', { name: /^(Maximize|Restore) queue$/ });
    const pressToggle = async () => {
      await act(() => toggle().element().focus());
      await userEvent.keyboard('{Enter}');
    };

    it('takes the detail column beside the library side by side, and gives it back', async () => {
      await render(WIDE_PX);
      await row('Item 2').click();
      const libraryBounds = libraryPane();

      await pressToggle();

      await expect.element(toggle()).toHaveTextContent('Restore queue');
      expect(footer()).toEqual({ bottom: grid().height, left: libraryBounds.right, right: grid().width, top: 0 });
      // The library stays as it was; the detail leaves the accessibility tree and tab order.
      expect(libraryPane()).toEqual(libraryBounds);
      await expect.element(library()).toBeVisible();
      await expect.element(detail()).not.toBeInTheDocument();
      draft().focus();
      expect(focused()).toBe(toggle().element());
      expect(await auditAccessibility(host)).toEqual([]);

      await pressToggle();

      await expect.element(detail()).toBeVisible();
      await expect.element(page.getByRole('heading', { name: 'item-2' })).toBeVisible();
      expect(focused()).toBe(toggle().element());
      expect(footer().top).toBeGreaterThan(grid().height / 2);
    });

    it('moves focus to the footer when a maximize requested elsewhere hides the control holding it', async () => {
      await render(WIDE_PX);
      await row('Item 2').click();
      await act(() => draft().focus());

      // As a "View queue" link or a hotkey would, not the footer's own control.
      await act(() =>
        toggle()
          .element()
          .dispatchEvent(new MouseEvent('click', { bubbles: true }))
      );

      await expect.element(detail()).not.toBeInTheDocument();
      expect(focused().closest('[data-manager-footer]')).not.toBeNull();
      expect(isShown(focused())).toBe(true);
    });

    it('takes the whole manager when the panes take turns, and gives it back', async () => {
      await render(NARROW_PX);

      await pressToggle();

      expect(footer()).toEqual({ bottom: grid().height, left: 0, right: grid().width, top: 0 });
      await expect.element(library()).not.toBeInTheDocument();
      await expect.element(detail()).not.toBeInTheDocument();
      host.querySelector<HTMLElement>('[data-manager-pane="library"] button')!.focus();
      expect(focused()).toBe(toggle().element());

      await pressToggle();

      await expect.element(library()).toBeVisible();
      expect(focused()).toBe(toggle().element());
    });
  });
});
