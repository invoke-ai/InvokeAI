import { Box, ChakraProvider, Icon, Menu, Portal } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { CopyIcon } from 'lucide-react';
import { Activity, act, useCallback, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, server, userEvent } from 'vitest/browser';

import { findArrowJointSeams } from './arrowJoint.testing';
import { MenuActionItem, MenuContent, MenuHeadingGroup, MenuIconItem, MenuItem, useContextMenu } from './Menu';

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
    await act(() => item('copy-path').click());

    expect(actions.copyImage).not.toHaveBeenCalled();
    expect(actions.copyJson).not.toHaveBeenCalled();
    expect(actions.copyPath).not.toHaveBeenCalled();
    expect(actions.rootSelect).not.toHaveBeenCalled();
    expect(item('rename')).toBeVisible();

    await act(() => userEvent.click(item('rename')));
    expect(actions.rename).toHaveBeenCalledOnce();
    expect(actions.rootSelect).toHaveBeenCalledWith({
      modifiers: { altKey: false, ctrlKey: false, metaKey: false, shiftKey: false },
      value: 'rename',
    });
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
    expect(headingRect.height).toBeLessThanOrEqual(groupLabel.getBoundingClientRect().height + 0.5);

    const nameHead = heading.querySelector<HTMLElement>('[data-full-text] > span')!;
    const state = Array.from(heading.children).at(-1) as HTMLElement;
    expect(state.textContent).toBe('Unsaved changes');
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

describe('menu item icons', () => {
  it("sit back from the label by default, keep a danger item's tone, and yield to an explicit color", async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <Menu.Root open>
            <Menu.Positioner>
              <MenuContent>
                <Menu.Item value="plain">
                  <CopyIcon data-testid="plain-icon" />
                  Copy
                </Menu.Item>
                <Menu.Item data-danger="" value="danger">
                  <CopyIcon data-testid="danger-icon" />
                  Delete
                </Menu.Item>
                <Menu.Item value="explicit">
                  <Icon as={CopyIcon} color="fg" data-testid="explicit-icon" />
                  Checked
                </Menu.Item>
                {/* Token references resolved in the same color context as the items. */}
                <Box color="fg.subtle" data-testid="subtle-reference" />
                <Box color="fg" data-testid="fg-reference" />
              </MenuContent>
            </Menu.Positioner>
          </Menu.Root>
        </ChakraProvider>
      );
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(item('plain')).toBeVisible());

    const color = (testId: string) => getComputedStyle(document.querySelector(`[data-testid="${testId}"]`)!).color;
    const subtle = color('subtle-reference');
    const fg = color('fg-reference');
    const danger = getComputedStyle(item('danger')).color;
    // Three distinct colors, so each expectation below tells the rules apart.
    expect(new Set([subtle, fg, danger]).size).toBe(3);

    expect(color('plain-icon')).toBe(subtle);
    expect(color('danger-icon')).toBe(danger);
    expect(color('explicit-icon')).toBe(fg);
  });
});

const ROOT_END = { placement: 'bottom-end' } as const;
const SUBMENU_RIGHT = { placement: 'right-start' } as const;

describe('submenus', () => {
  it('flipped to open left keep their placement and open animation while the pointer crosses into them', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          {/* At the viewport's right edge, so a right-start submenu has no room and flips to the left. */}
          <Box position="fixed" right="1rem" top="2rem">
            <Menu.Root positioning={ROOT_END}>
              <Menu.Trigger>Open</Menu.Trigger>
              <Menu.Positioner>
                <MenuContent minW="12rem">
                  <Menu.Item value="rename">Rename</Menu.Item>
                  {/* As call sites nest them: right-start, positioned in a portal. */}
                  <Menu.Root positioning={SUBMENU_RIGHT}>
                    <Menu.TriggerItem>Move to</Menu.TriggerItem>
                    <Portal>
                      <Menu.Positioner>
                        <MenuContent minW="12rem">
                          <Menu.Item value="inbox">Inbox</Menu.Item>
                          <Menu.Item value="archive">Archive</Menu.Item>
                        </MenuContent>
                      </Menu.Positioner>
                    </Portal>
                  </Menu.Root>
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

    const trigger = page.getByRole('menuitem', { name: 'Move to' }).element() as HTMLElement;
    await act(() => userEvent.hover(trigger));
    await vi.waitFor(() => expect(item('inbox')).toBeVisible());
    const submenu = item('inbox').closest<HTMLElement>('[role="menu"]')!;
    await vi.waitFor(() => expect(submenu.dataset.placement).toMatch(/^left/));
    await Promise.all(submenu.getAnimations().map((animation) => animation.finished));

    const changes: string[] = [];
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        const element = record.target as HTMLElement;
        const name = record.attributeName!;
        changes.push(`${element === trigger ? 'trigger' : 'submenu'} ${name}=${String(element.getAttribute(name))}`);
      }
    });
    for (const target of [trigger, submenu]) {
      observer.observe(target, { attributeFilter: ['data-placement', 'data-state'] });
    }
    const animationStarts: string[] = [];
    const recordAnimation = (event: AnimationEvent) => {
      if (event.target === submenu) {
        animationStarts.push(event.animationName);
      }
    };
    submenu.addEventListener('animationstart', recordAnimation);

    // Real pointer moves a pixel apart, as a hand crosses; `userEvent.hover` jumps, and force skips stability waits.
    const body = document.body.getBoundingClientRect();
    const from = trigger.getBoundingClientRect();
    const y = from.top + from.height / 2;
    const end = submenu.getBoundingClientRect().right - 16;
    expect(end).toBeLessThan(from.left);
    for (let x = from.left + from.width / 2; x >= end; x -= 1) {
      await act(() =>
        userEvent.hover(page.elementLocator(document.body), {
          force: true,
          position: { x: x - body.left, y: y - body.top },
        })
      );
    }
    // Long enough for a restarted open animation, and for the hover the submenu loses with it to come back.
    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 300);
    });
    observer.disconnect();
    submenu.removeEventListener('animationstart', recordAnimation);

    expect(document.elementFromPoint(end, y)?.closest('[role="menu"]')).toBe(submenu);
    expect({ animationStarts, changes, state: submenu.dataset.state }).toEqual({
      animationStarts: [],
      changes: [],
      state: 'open',
    });
    expect(submenu.dataset.placement).toMatch(/^left/);
  });
});

const ContextMenuAt = () => {
  const { anchor, close, onRequestDismiss, open } = useContextMenu();
  const openLow = useCallback(() => open({ x: 120, y: window.innerHeight - 4 }), [open]);
  const openHigh = useCallback(() => open({ x: 120, y: 60 }), [open]);
  const positioning = useMemo(
    () => ({
      getAnchorRect: () => (anchor ? { height: 1, width: 1, x: anchor.x, y: anchor.y } : null),
      placement: 'bottom-start' as const,
    }),
    [anchor]
  );
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        close();
      }
    },
    [close]
  );
  return (
    <>
      <button type="button" onClick={openLow}>
        Open low
      </button>
      <button type="button" onClick={openHigh}>
        Open high
      </button>
      <Menu.Root
        open={anchor !== null}
        positioning={positioning}
        onOpenChange={handleOpenChange}
        onRequestDismiss={onRequestDismiss}
      >
        <Menu.Positioner>
          <MenuContent minW="12rem">
            <Menu.Item value="rename">Rename</Menu.Item>
            <Menu.Item value="duplicate">Duplicate</Menu.Item>
          </MenuContent>
        </Menu.Positioner>
      </Menu.Root>
    </>
  );
};

describe('reopened menus', () => {
  it('open from their configured side, without a frame on the side a previous opening flipped to', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <ContextMenuAt />
        </ChakraProvider>
      );
      await Promise.resolve();
    });

    await act(() => userEvent.click(page.getByRole('button', { name: 'Open low' })));
    await vi.waitFor(() => expect(item('rename')).toBeVisible());
    const content = item('rename').closest<HTMLElement>('[role="menu"]')!;
    await vi.waitFor(() => expect(content.dataset.placement).toBe('top-start'));
    await Promise.all(content.getAnimations().map((animation) => animation.finished));

    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => expect(content.dataset.state).toBe('closed'));
    await Promise.all(content.getAnimations().map((animation) => animation.finished));

    // The content element may be replaced, so watch whichever one is current.
    const currentContent = () => document.querySelector<HTMLElement>('[data-scope="menu"][data-part="content"]');
    const snapshots: string[] = [];
    const observer = new MutationObserver(() => {
      const current = currentContent();
      snapshots.push(`${String(current?.dataset.state)} ${String(current?.dataset.placement)}`);
    });
    observer.observe(document.body, {
      attributeFilter: ['data-placement', 'data-state'],
      childList: true,
      subtree: true,
    });
    // One opening starts its animations together, in one frame; a restarted one starts in a later frame.
    const animationStarts: { name: string; time: CSSNumberish | null }[] = [];
    const recordAnimation = (event: AnimationEvent) => {
      if (event.target === currentContent()) {
        animationStarts.push({ name: event.animationName, time: document.timeline.currentTime });
      }
    };
    document.addEventListener('animationstart', recordAnimation, true);

    await act(() => userEvent.click(page.getByRole('button', { name: 'Open high' })));
    await vi.waitFor(() => expect(currentContent()?.dataset.state).toBe('open'));
    await vi.waitFor(() => expect(currentContent()?.dataset.placement).toBe('bottom-start'));
    const reopened = currentContent()!;
    await Promise.all(reopened.getAnimations().map((animation) => animation.finished));
    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 200);
    });
    observer.disconnect();
    document.removeEventListener('animationstart', recordAnimation, true);

    expect(animationStarts.length).toBeGreaterThan(0);
    expect({
      openOnFlippedSide: snapshots.filter((snapshot) => snapshot.startsWith('open top')),
      startedLater: animationStarts.filter(({ time }) => time !== animationStarts[0]!.time).map(({ name }) => name),
    }).toEqual({ openOnFlippedSide: [], startedLater: [] });
  });
});

const ARROW_PLACEMENTS = {
  'bottom-start': { placement: 'bottom-start' },
  'top-end': { placement: 'top-end' },
  'right-start': { placement: 'right-start' },
  'left-start': { placement: 'left-start' },
} as const;

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
          <Box
            left={placement.startsWith('right') ? '1rem' : placement.startsWith('left') ? 'calc(100vw - 5rem)' : '50vw'}
            position="fixed"
            top="50vh"
          >
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

    const contentStyle = getComputedStyle(menu);
    const tipStyle = getComputedStyle(tip);
    expect(tipStyle.backgroundColor).toBe(contentStyle.backgroundColor);
    expect(tipStyle.borderTopColor).toBe(contentStyle.borderTopColor);
    expect(tipStyle.borderTopWidth).toBe(contentStyle.borderTopWidth);
  });

  it('stays open, with keyboard navigation intact, when its tip is clicked; a click beside the menu still closes it', async () => {
    await renderTriggerMenu('bottom-start', true);
    const trigger = page.getByRole('button', { name: 'Open' }).element();
    const menu = page.getByRole('menu').element();
    const [arrow] = arrowsIn(menu.parentElement);
    await vi.waitFor(() => expect(menu.dataset.side).toBe('bottom'));
    await Promise.all([...menu.getAnimations(), ...arrow!.getAnimations()].map((animation) => animation.finished));
    const tipBox = arrow!.querySelector<HTMLElement>('[data-part="arrow-tip"]')!.getBoundingClientRect();
    const content = menu.getBoundingClientRect();
    const body = document.body.getBoundingClientRect();
    const clickAt = (x: number, y: number) =>
      act(() =>
        userEvent.click(page.elementLocator(document.body), { position: { x: x - body.left, y: y - body.top } })
      );
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await vi.waitFor(() => expect(item('rename').hasAttribute('data-highlighted')).toBe(true));

    await clickAt((tipBox.left + tipBox.right) / 2, content.top - 3);
    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 100);
    });
    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    expect(menu.contains(document.activeElement)).toBe(true);
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await vi.waitFor(() =>
      expect(page.getByRole('menuitem', { name: 'Move to' }).element()).toHaveAttribute('data-highlighted')
    );

    const arrowBox = arrow!.getBoundingClientRect();
    const clearance = { x: tipBox.right + 2, y: content.top - 3 };
    expect(clearance.x).toBeLessThan(arrowBox.right);
    expect(arrow!.contains(document.elementFromPoint(clearance.x, clearance.y))).toBe(false);
    await clickAt(clearance.x, clearance.y);
    await vi.waitFor(() => expect(trigger.getAttribute('aria-expanded')).toBe('false'));
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
    await vi.waitFor(() => {
      expect(arrow!.querySelector<HTMLElement>('[data-part="arrow-tip"]')!.offsetWidth).toBe(arrowSize);
      const rect = getAnchorRect();
      expect(menu.getBoundingClientRect().top - (rect.y + rect.height)).toBeCloseTo(gap, 0);
    });
    if (arrowSize === 0) {
      // Floating UI keeps even a zero-size arrow 4px from the corner, which would shift the menu.
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
    const y = placement === 'bottom-start' ? content.top + 9 : content.bottom - 9;
    expect(y).toBeGreaterThan(box.top);
    expect(y).toBeLessThan(box.bottom);

    expect(document.elementFromPoint(box.left + box.width / 2, y)?.closest('[role="menuitem"]')).toBe(item('rename'));
  });

  describe('joint with the content', () => {
    afterEach(async () => {
      await server.commands.emulateDevicePixelRatio(null);
      applyThemeToRoot('classic');
    });

    it.each([
      ['classic', 1, 'top-end'],
      ['classic', 1, 'bottom-start'],
      ['classic', 1.25, 'top-end'],
      ['classic', 1.25, 'bottom-start'],
      ['classic', 1.25, 'right-start'],
      ['classic', 1.25, 'left-start'],
      ['light', 1.25, 'top-end'],
    ] as const)(
      'reads as one shape, the content border never crossing the arrow (%s theme, DPR %s, %s)',
      async (themeId, ratio, placement) => {
        applyThemeToRoot(themeId);
        await server.commands.emulateDevicePixelRatio(ratio);
        await vi.waitFor(() => expect(globalThis.devicePixelRatio).toBe(ratio));
        await renderTriggerMenu(placement);
        const menu = page.getByRole('menu').element() as HTMLElement;
        const [arrow] = arrowsIn(menu.parentElement);
        await vi.waitFor(() => expect(menu.dataset.placement).toBe(placement));
        await Promise.all([...menu.getAnimations(), ...arrow!.getAnimations()].map((animation) => animation.finished));

        expect(await findArrowJointSeams(menu, arrow!.querySelector<HTMLElement>('[data-part="arrow-tip"]')!)).toEqual(
          []
        );
      }
    );
  });
});

const MenuSections = () => {
  const [isOtherShown, setIsOtherShown] = useState(false);
  const showOther = useCallback(() => setIsOtherShown(true), []);
  const showFirst = useCallback(() => setIsOtherShown(false), []);

  return (
    <ChakraProvider value={system}>
      <button type="button" onClick={showFirst}>
        First section
      </button>
      <button type="button" onClick={showOther}>
        Other section
      </button>
      <Activity mode={isOtherShown ? 'hidden' : 'visible'}>
        <Menu.Root>
          <Menu.Trigger>Actions</Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent>
                <MenuItem value="rename">Rename</MenuItem>
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Activity>
    </ChakraProvider>
  );
};

describe('menus in a hidden section', () => {
  it('stay closed when their section returns after it was hidden while they faded out', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => root?.render(<MenuSections />));
    const shownMenu = () =>
      [...document.querySelectorAll<HTMLElement>('[data-scope="menu"][data-part="content"]')].find((menu) =>
        menu.checkVisibility()
      );

    await userEvent.click(page.getByRole('button', { name: 'Actions' }));
    await expect.poll(() => shownMenu()?.textContent).toBe('Rename');
    await act(() => userEvent.keyboard('{Escape}'));
    await act(() => userEvent.click(page.getByRole('button', { name: 'Other section' })));
    await act(() => userEvent.click(page.getByRole('button', { name: 'First section' })));

    await expect.poll(shownMenu).toBeUndefined();
    await userEvent.click(page.getByRole('button', { name: 'Actions' }));
    await expect.poll(() => shownMenu()?.textContent).toBe('Rename');
  });
});
