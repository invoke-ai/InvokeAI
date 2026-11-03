import { Box, ChakraProvider, Menu } from '@chakra-ui/react';
import { system } from '@theme/system';
import { CopyIcon } from 'lucide-react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { MenuActionItem, MenuContent, MenuHeadingGroup, MenuIconItem, MenuItem } from './Menu';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const REASON = 'Copying needs HTTPS or localhost';

const handlers = () => ({
  copyImage: vi.fn(),
  copyJson: vi.fn(),
  copyPath: vi.fn(),
  rename: vi.fn(),
  rootSelect: vi.fn(),
});

const renderMenu = async (actions: ReturnType<typeof handlers>) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <Menu.Root onSelect={actions.rootSelect}>
          <Menu.Trigger>Actions</Menu.Trigger>
          <Menu.Positioner>
            <Menu.Content>
              <MenuIconItem
                icon={CopyIcon}
                label="Copy image"
                unavailableReason={REASON}
                value="copy-image"
                onSelect={actions.copyImage}
              />
              <MenuActionItem label="Rename" value="rename" onSelect={actions.rename} />
              {/* The caller's own `disabled` must not hide the reason. */}
              <MenuActionItem
                disabled
                label="Copy JSON"
                unavailableReason={REASON}
                value="copy-json"
                onSelect={actions.copyJson}
              />
              <MenuItem
                aria-describedby="copy-path-hint"
                unavailableReason={REASON}
                value="copy-path"
                onClick={actions.copyPath}
              >
                <CopyIcon />
                <Menu.ItemText>Copy path</Menu.ItemText>
              </MenuItem>
            </Menu.Content>
          </Menu.Positioner>
        </Menu.Root>
        <p id="copy-path-hint">Copies the folder path</p>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
};

const item = (value: string): HTMLElement => document.querySelector<HTMLElement>(`[data-value="${value}"]`)!;
const tooltipText = (): string | null | undefined => document.querySelector('[role="tooltip"]')?.textContent;

const highlight = async (value: string) => {
  await act(() => userEvent.keyboard('{ArrowDown}'));
  await vi.waitFor(() => expect(item(value).hasAttribute('data-highlighted')).toBe(true));
};

describe('unavailable menu items', () => {
  it('stay reachable by keyboard, read as disabled and described by their reason, and never act', async () => {
    const actions = handlers();
    await renderMenu(actions);

    await act(() => userEvent.click(page.getByRole('button', { name: 'Actions' })));
    await vi.waitFor(() => expect(item('rename')).toBeVisible());

    // Their name stays the label; the reason is their description, read without the tooltip, after the caller's own.
    for (const [value, name, description] of [
      ['copy-image', 'Copy image', REASON],
      ['copy-json', 'Copy JSON', REASON],
      ['copy-path', 'Copy path', `Copies the folder path ${REASON}`],
    ] as const) {
      expect(item(value).getAttribute('aria-disabled')).toBe('true');
      // One description mechanism: `aria-description` beside `aria-describedby` would be read twice by some readers.
      expect(item(value).hasAttribute('aria-description')).toBe(false);
      const menuItem = page.getByRole('menuitem', { name, exact: true });
      await expect.element(menuItem).toHaveAccessibleName(name);
      await expect.element(menuItem).toHaveAccessibleDescription(description);
    }

    // Keyboard navigation does not skip them, and highlighting shows the reason without a pointer. The icon-only
    // item shows its label beside the reason, since the tooltip is the only place its name appears.
    await highlight('copy-image');
    await vi.waitFor(() => expect(tooltipText()).toBe(`Copy image${REASON}`));
    await act(() => userEvent.keyboard('{Enter}'));

    await highlight('rename');
    await highlight('copy-json');
    await vi.waitFor(() => expect(tooltipText()).toBe(REASON));
    await act(() => userEvent.keyboard('{Enter}'));

    await highlight('copy-path');
    await vi.waitFor(() => expect(tooltipText()).toBe(REASON));
    await act(() => userEvent.keyboard('{Enter}'));
    // A pointer click on an unavailable item does nothing either.
    await act(() => item('copy-path').click());

    expect(actions.copyImage).not.toHaveBeenCalled();
    expect(actions.copyJson).not.toHaveBeenCalled();
    expect(actions.copyPath).not.toHaveBeenCalled();
    expect(actions.rootSelect).not.toHaveBeenCalled();
    // Selecting an unavailable item leaves the menu open, as a disabled one would.
    expect(item('rename')).toBeVisible();

    // The control: an available item's selection reaches both its handler and the menu root.
    await act(() => userEvent.click(item('rename')));
    expect(actions.rename).toHaveBeenCalledOnce();
    expect(actions.rootSelect).toHaveBeenCalledWith({ value: 'rename' });
  });
});

describe('menu heading groups', () => {
  const LONG_NAME = 'A preset name far too long for the menu to show whole';

  it('title the menu at group-label height, truncating the name beside its state, without a focusable row', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <Menu.Root open>
            <Menu.Positioner>
              <Menu.Content w="12rem">
                <MenuHeadingGroup label={LONG_NAME} trailing="Unsaved changes">
                  <Menu.Item value="save">Save changes</Menu.Item>
                </MenuHeadingGroup>
                <Menu.ItemGroup>
                  <Menu.ItemGroupLabel data-testid="group-label">Group</Menu.ItemGroupLabel>
                  <Menu.Item value="other">Other</Menu.Item>
                </Menu.ItemGroup>
              </Menu.Content>
            </Menu.Positioner>
          </Menu.Root>
        </ChakraProvider>
      );
      await Promise.resolve();
    });

    const group = page.getByRole('group', { name: `${LONG_NAME} Unsaved changes` });
    await expect.element(group).toBeVisible();
    const heading = group.element().querySelector<HTMLElement>('[data-part="item-group-label"]')!;
    const groupLabel = document.querySelector<HTMLElement>('[data-testid="group-label"]')!;
    const headingRect = heading.getBoundingClientRect();
    // The oversized title band this replaced stood a third taller than a group label.
    expect(headingRect.height).toBeLessThanOrEqual(groupLabel.getBoundingClientRect().height + 0.5);

    const nameHead = heading.querySelector<HTMLElement>('[data-full-text] > span')!;
    const state = Array.from(heading.children).at(-1) as HTMLElement;
    expect(state.textContent).toBe('Unsaved changes');
    // The name gives way; the state keeps its whole width on the heading's row.
    expect(nameHead.clientWidth).toBeLessThan(nameHead.scrollWidth);
    expect(state.getBoundingClientRect().right).toBeLessThanOrEqual(headingRect.right);
    expect(state.getBoundingClientRect().top).toBeGreaterThanOrEqual(headingRect.top);
    expect(state.getBoundingClientRect().bottom).toBeLessThanOrEqual(headingRect.bottom);

    expect(
      page
        .getByRole('menuitem')
        .elements()
        .map((element) => element.textContent)
    ).toEqual(['Save changes', 'Other']);
  });
});

const ARROW_PLACEMENTS = {
  'bottom-start': { placement: 'bottom-start' },
  'top-end': { placement: 'top-end' },
  'right-start': { placement: 'right-start' },
} as const;

// Context menus open at the pointer, anchored to a 1px point; other measured anchors are boxes, such as a tab.
const POINT_START = { height: 1, width: 1, x: 40, y: 200 };
const POINT_END = { height: 1, width: 1, x: 300, y: 200 };
const BOX = { height: 24, width: 40, x: 40, y: 200 };
const ANCHORED = {
  'point-start': { getAnchorRect: () => POINT_START, placement: 'bottom-start' },
  'point-end': { getAnchorRect: () => POINT_END, placement: 'bottom-end' },
  box: { getAnchorRect: () => BOX, placement: 'bottom-start' },
} as const;
const END_ALIGNED = { placement: 'bottom-end' } as const;

describe('menu arrows', () => {
  const renderTriggerMenu = async (placement: keyof typeof ARROW_PLACEMENTS, submenu = false) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          {/* Room on the side the menu opens to, so it does not flip. */}
          <Box left={placement.startsWith('right') ? '1rem' : '50vw'} position="fixed" top="50vh">
            <Menu.Root positioning={ARROW_PLACEMENTS[placement]}>
              <Menu.Trigger>Open</Menu.Trigger>
              <Menu.Positioner>
                <MenuContent minW="12rem">
                  <Menu.Item value="rename">Rename</Menu.Item>
                  {submenu ? (
                    <Menu.Root>
                      <Menu.TriggerItem>Move to</Menu.TriggerItem>
                      <Menu.Positioner>
                        <MenuContent>
                          <Menu.Item value="inbox">Inbox</Menu.Item>
                        </MenuContent>
                      </Menu.Positioner>
                    </Menu.Root>
                  ) : null}
                </MenuContent>
              </Menu.Positioner>
            </Menu.Root>
          </Box>
        </ChakraProvider>
      );
      await Promise.resolve();
    });
    await act(() => userEvent.click(page.getByRole('button', { name: 'Open' })));
    await vi.waitFor(() => expect(item('rename')).toBeVisible());
  };

  const arrowsIn = (positioner: Element | null | undefined) =>
    positioner ? [...positioner.querySelectorAll<HTMLElement>('[data-scope="menu"][data-part="arrow"]')] : [];

  it.each([
    ['bottom-start', 'top'],
    ['top-end', 'bottom'],
    ['right-start', 'left'],
  ] as const)('points a %s menu at its trigger from the content edge facing it', async (placement, edge) => {
    await renderTriggerMenu(placement);
    const trigger = page.getByRole('button', { name: 'Open' }).element().getBoundingClientRect();
    const menu = page.getByRole('menu').element();
    const [arrow] = arrowsIn(menu.parentElement);
    const tip = arrow!.querySelector<HTMLElement>('[data-part="arrow-tip"]')!;
    // The arrow slides in with the content once placed; measure both at rest.
    await vi.waitFor(() => expect(menu.dataset.side).toBe(placement.split('-')[0]));
    await Promise.all([...menu.getAnimations(), ...arrow!.getAnimations()].map((animation) => animation.finished));

    // Beside the content, whose scrolling would clip it, not inside it.
    expect(menu.contains(arrow!)).toBe(false);
    const content = menu.getBoundingClientRect();
    const tipBox = tip.getBoundingClientRect();
    if (edge === 'left') {
      expect(tipBox.left).toBeLessThan(content.left);
      expect(tipBox.left).toBeGreaterThanOrEqual(trigger.right);
      expect((tipBox.top + tipBox.bottom) / 2).toBeCloseTo((trigger.top + trigger.bottom) / 2, 0);
    } else {
      if (edge === 'top') {
        expect(tipBox.top).toBeLessThan(content.top);
        expect(tipBox.top).toBeGreaterThanOrEqual(trigger.bottom);
      } else {
        expect(tipBox.bottom).toBeGreaterThan(content.bottom);
        expect(tipBox.bottom).toBeLessThanOrEqual(trigger.top);
      }
      expect((tipBox.left + tipBox.right) / 2).toBeCloseTo((trigger.left + trigger.right) / 2, 0);
    }

    // The tip reads as part of the content: its fill and border are the content's, not a transparent outline.
    const contentStyle = getComputedStyle(menu);
    const tipStyle = getComputedStyle(tip);
    expect(tipStyle.backgroundColor).toBe(contentStyle.backgroundColor);
    expect(tipStyle.borderTopColor).toBe(contentStyle.borderTopColor);
    expect(tipStyle.borderTopWidth).toBe(contentStyle.borderTopWidth);
  });

  it('leaves submenus, which open flush beside their item, without one', async () => {
    await renderTriggerMenu('bottom-start', true);
    await act(() => userEvent.click(page.getByRole('menuitem', { name: 'Move to' })));
    await vi.waitFor(() => expect(item('inbox')).toBeVisible());

    const [parentMenu, submenu] = page.getByRole('menu').elements();
    expect(arrowsIn(parentMenu!.parentElement)).toHaveLength(1);
    expect(arrowsIn(submenu!.parentElement)).toHaveLength(0);
  });

  it.each([
    ['a bottom-start context menu has none, and opens right at the pointer', 'point-start', 0, 8],
    ['a bottom-end context menu has none, and opens right at the pointer', 'point-end', 0, 8],
    ['a menu anchored to a measured box points at it', 'box', 12, 8 + 6],
  ] as const)('%s', async (_name, anchor, arrowSize, gap) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <Menu.Root open positioning={ANCHORED[anchor]}>
            <Menu.Positioner>
              <MenuContent>
                <Menu.Item value="rename">Rename</Menu.Item>
              </MenuContent>
            </Menu.Positioner>
          </Menu.Root>
        </ChakraProvider>
      );
      await Promise.resolve();
    });
    const menu = page.getByRole('menu').element();
    const [arrow] = arrowsIn(menu.parentElement);
    const { getAnchorRect, placement } = ANCHORED[anchor];
    await vi.waitFor(() => expect(menu.dataset.placement).toBe(placement));
    // A point anchor shrinks the arrow once placed; the menu is then placed again without the arrow's height.
    await vi.waitFor(() => {
      expect(arrow!.querySelector<HTMLElement>('[data-part="arrow-tip"]')!.offsetWidth).toBe(arrowSize);
      const rect = getAnchorRect();
      expect(menu.getBoundingClientRect().top - (rect.y + rect.height)).toBeCloseTo(gap, 0);
    });
    if (arrowSize === 0) {
      // Nor does its tip leave a bordered dot, nor its box shift the menu sideways to centre on the pointer: the menu's
      // aligned edge is at the pointer. Floating UI keeps even an arrow of no size its 4px padding from the corner, which
      // moves the menu by up to that, as it did before menus had arrows.
      expect(getComputedStyle(arrow!.querySelector('[data-part="arrow-tip"]')!).borderTopWidth).toBe('0px');
      const rect = getAnchorRect();
      const content = menu.getBoundingClientRect();
      if (placement === 'bottom-start') {
        expect(Math.abs(content.left - rect.x)).toBeLessThanOrEqual(4);
      } else {
        expect(Math.abs(content.right - (rect.x + rect.width))).toBeLessThanOrEqual(4);
      }
    } else {
      const tip = arrow!.querySelector('[data-part="arrow-tip"]')!.getBoundingClientRect();
      expect((tip.left + tip.right) / 2).toBeCloseTo(BOX.x + BOX.width / 2, 0);
    }
  });

  it.each([
    ['clears the corner for a small trigger at the end of a bar, shifting the panel to point at it', '50vw'],
    ['clears the corner for a small trigger at the viewport edge, where the panel cannot shift', '0px'],
  ] as const)('%s', async (_name, right) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <Box position="fixed" right={right} top="40vh">
            <Menu.Root positioning={END_ALIGNED}>
              <Menu.Trigger aria-label="More" h="4" p="0" w="4" />
              <Menu.Positioner>
                <MenuContent minW="12rem">
                  <Menu.Item value="rename">Rename</Menu.Item>
                </MenuContent>
              </Menu.Positioner>
            </Menu.Root>
          </Box>
        </ChakraProvider>
      );
      await Promise.resolve();
    });
    await act(() => userEvent.click(page.getByRole('button', { name: 'More' })));
    const menu = page.getByRole('menu').element();
    await vi.waitFor(() => expect(menu.dataset.side).toBe('bottom'));
    await Promise.all(menu.getAnimations().map((animation) => animation.finished));

    const content = menu.getBoundingClientRect();
    const radius = parseFloat(getComputedStyle(menu).borderTopRightRadius);
    const tip = arrowsIn(menu.parentElement)[0]!.querySelector('[data-part="arrow-tip"]')!.getBoundingClientRect();
    // The diamond's widest extent is its base on the content's edge: it stays on the straight part of that edge.
    expect(radius).toBeGreaterThan(0);
    expect(tip.left).toBeGreaterThanOrEqual(content.left + radius);
    expect(tip.right).toBeLessThanOrEqual(content.right - radius);
    if (right !== '0px') {
      const trigger = page.getByRole('button', { name: 'More' }).element().getBoundingClientRect();
      expect((tip.left + tip.right) / 2).toBeCloseTo((trigger.left + trigger.right) / 2, 0);
    }
  });

  it.each([
    ['bottom-start', 'first item, below the arrow'],
    ['top-end', 'last item, above the arrow'],
  ] as const)('lets the pointer through the arrow box to a %s menu: its %s', async (placement, _where) => {
    await renderTriggerMenu(placement);
    const menu = page.getByRole('menu').element();
    await vi.waitFor(() => expect(menu.dataset.side).toBe(placement.split('-')[0]));
    await Promise.all(menu.getAnimations().map((animation) => animation.finished));
    const [arrow] = arrowsIn(menu.parentElement);
    const box = arrow!.getBoundingClientRect();
    const content = menu.getBoundingClientRect();
    // Inside the item, which starts past the content's padding, and inside the arrow box's half over the content.
    const y = placement === 'bottom-start' ? content.top + 9 : content.bottom - 9;
    expect(y).toBeGreaterThan(box.top);
    expect(y).toBeLessThan(box.bottom);

    expect(document.elementFromPoint(box.left + box.width / 2, y)?.closest('[role="menuitem"]')).toBe(item('rename'));
  });
});
