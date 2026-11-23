import { ChakraProvider, Menu, Popover, Portal } from '@chakra-ui/react';
import { system } from '@theme/system';
import { CameraIcon } from 'lucide-react';
import { Activity, act, type ReactElement, useCallback, useRef, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { MenuContent } from './Menu';
import { PopoverContent } from './Popover';
import { Toolbar, ToolbarButton } from './Toolbar';
import { ClippedTooltip, Tooltip } from './Tooltip';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** A one-shot action that is busy, and so disabled, while it runs. */
const BusyAction = () => {
  const [isBusy, setIsBusy] = useState(false);
  const start = useCallback(() => setIsBusy(true), []);
  return (
    <Toolbar left="200px" position="absolute" top="200px">
      <ToolbarButton disabled={isBusy} icon={CameraIcon} label="Export" loading={isBusy} onClick={start} />
    </Toolbar>
  );
};

/** Runs `run` while animation frames are held back, as when it all happens within one frame. */
const withinOneFrame = async (run: () => Promise<void>): Promise<void> => {
  const { cancelAnimationFrame, requestAnimationFrame } = window;
  let nextId = 1_000_000;
  window.requestAnimationFrame = () => (nextId += 1);
  window.cancelAnimationFrame = () => undefined;
  try {
    await run();
  } finally {
    window.requestAnimationFrame = requestAnimationFrame;
    window.cancelAnimationFrame = cancelAnimationFrame;
  }
};

const exportTooltip = () =>
  [...document.querySelectorAll<HTMLElement>('[data-scope="tooltip"][data-part="content"]')].find(
    (candidate) => candidate.textContent === 'Export'
  );

it('keeps a tooltip that closes before it was placed out of sight while it fades', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <BusyAction />
      </ChakraProvider>
    )
  );
  const button = host.querySelector<HTMLButtonElement>('button[aria-label="Export"]')!;

  // Keyboard focus opens the tooltip, which is placed a frame later; activating the button first closes it.
  await withinOneFrame(async () => {
    // Async act lets the tooltip render open; frames stay held, so it is not placed yet.
    await act(() => {
      document.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Tab' }));
      button.focus();
      return Promise.resolve();
    });
    expect(exportTooltip()?.dataset.state).toBe('open');
    await act(() => {
      button.click();
      return Promise.resolve();
    });
  });

  expect(button.disabled).toBe(true);
  // Still mounted for its exit animation; it must not fade out at the page's top-left corner.
  expect(exportTooltip()?.dataset.state).toBe('closed');
  expect(exportTooltip()!.getBoundingClientRect().bottom).toBeLessThanOrEqual(0);
});

/** A name in a button of the given width that truncates whatever does not fit. */
const NameButton = ({ name, width }: { name: string; width: string }) => {
  const ref = useRef<HTMLButtonElement>(null);
  const isClipped = useCallback(() => !!ref.current && ref.current.scrollWidth > ref.current.clientWidth, []);
  return (
    <ClippedTooltip content={name} isClipped={isClipped}>
      <button
        ref={ref}
        style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', width }}
        type="button"
      >
        {name}
      </button>
    </ClippedTooltip>
  );
};

const focusWithKeyboard = (button: HTMLButtonElement) =>
  act(() => {
    document.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Tab' }));
    button.focus();
    return Promise.resolve();
  });

const openTooltipText = () => document.querySelector('[role="tooltip"][data-state="open"]')?.textContent;

it('shows a clipped name on keyboard focus and nothing for a name that fits', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <NameButton name="A name much too long for its button" width="60px" />
        <NameButton name="Fits" width="200px" />
      </ChakraProvider>
    )
  );
  const [clipped, fitting] = host.querySelectorAll('button');

  await focusWithKeyboard(clipped!);
  expect(openTooltipText()).toBe('A name much too long for its button');

  await focusWithKeyboard(fitting!);
  await act(async () => {
    await new Promise((resolve) => {
      setTimeout(resolve, 200);
    });
  });
  expect(openTooltipText()).toBeUndefined();
});

it('lets a click through an open tip to the control under it, such as the next row of a dense list', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onNextClick = vi.fn();
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <div style={{ display: 'grid', gap: '2px', left: '200px', position: 'absolute', top: '200px' }}>
          <Tooltip content="Delete" openDelay={0} placement="bottom">
            <button style={{ height: '28px', width: '28px' }} type="button">
              A
            </button>
          </Tooltip>
          <button aria-label="Next" style={{ height: '28px', width: '28px' }} type="button" onClick={onNextClick}>
            B
          </button>
        </div>
      </ChakraProvider>
    )
  );
  const [first, next] = host.querySelectorAll('button');

  await focusWithKeyboard(first!);
  await expect.poll(openTooltipText).toBe('Delete');
  const target = next!.getBoundingClientRect();
  const centre = target.top + target.height / 2;
  const tip = () => document.querySelector('[role="tooltip"][data-state="open"]')!.getBoundingClientRect();
  // Once placed, the tip covers the next button's centre: the case this guards.
  await expect.poll(() => tip().top < centre && tip().bottom > centre).toBe(true);

  await userEvent.click(next!);

  expect(onNextClick).toHaveBeenCalledOnce();
});

/** Two sections kept alive by Activity, like the launchpad's pages: the tooltipped button switches to the other one. */
const Sections = () => {
  const [isOtherShown, setIsOtherShown] = useState(false);
  const showOther = useCallback(() => setIsOtherShown(true), []);
  const showFirst = useCallback(() => setIsOtherShown(false), []);

  return (
    <>
      <button type="button" onClick={showFirst}>
        First section
      </button>
      <Activity mode={isOtherShown ? 'hidden' : 'visible'}>
        <Tooltip content="Open the other section">
          <button aria-label="Other section" type="button" onClick={showOther}>
            Go
          </button>
        </Tooltip>
      </Activity>
    </>
  );
};

it('shows a tip again only when hovered after its section was hidden mid-fade', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <Sections />
      </ChakraProvider>
    )
  );
  const shownTip = () =>
    [...document.querySelectorAll<HTMLElement>('[role="tooltip"]')].find((tip) => tip.checkVisibility());
  const trigger = host.querySelector<HTMLButtonElement>('button[aria-label="Other section"]')!;

  await userEvent.hover(trigger);
  await expect.poll(() => shownTip()?.textContent).toBe('Open the other section');
  // The press closes the tip and the click hides its section before the fade ends.
  await act(() => userEvent.click(trigger));
  await act(() => userEvent.click(host!.querySelector('button')!));

  await expect.poll(shownTip).toBeUndefined();
  // The pointer left while the section was hidden; the next hover still opens the tip.
  await userEvent.hover(trigger);
  await expect.poll(() => shownTip()?.textContent).toBe('Open the other section');
});

const POPOVER_AT = { getAnchorRect: () => ({ height: 1, width: 1, x: 24, y: 24 }), placement: 'bottom-start' } as const;

// Shared by the tooltip and the menu, as `useTooltipTriggerIds` does: Zag finds each trigger by its id.
const FIELD_ID = 'media-field';
const TIP_IDS = { trigger: FIELD_ID };
const MENU_IDS = { contextTrigger: FIELD_ID };

interface MediaFieldProps {
  inPopover: boolean;
  /** Tooltip around the menu's trigger is the `useTooltipTriggerIds` order; dialogs need the menu's outside. */
  isMenuTriggerOutside: boolean;
}

/** A field that shows its name on hover and opens a context menu from the same element; the item's tip opens with it. */
const MediaFieldWithMenu = ({ inPopover, isMenuTriggerOutside }: MediaFieldProps) => {
  const [isNameShown, setIsNameShown] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const showName = useCallback(() => setIsNameShown(true), []);
  const handleMenuOpenChange = useCallback((details: { open: boolean }) => setIsMenuOpen(details.open), []);
  const button = (
    <button type="button" onPointerEnter={showName}>
      Media
    </button>
  );
  const nameTip = (trigger: ReactElement) => (
    <Tooltip content="Media name" ids={TIP_IDS} open={isNameShown} placement="bottom">
      {trigger}
    </Tooltip>
  );
  const field = (
    <Menu.Root ids={MENU_IDS} open={isMenuOpen} onOpenChange={handleMenuOpenChange}>
      {isMenuTriggerOutside ? (
        <Menu.ContextTrigger asChild>{nameTip(button)}</Menu.ContextTrigger>
      ) : (
        nameTip(<Menu.ContextTrigger asChild>{button}</Menu.ContextTrigger>)
      )}
      <Portal>
        <Menu.Positioner>
          <MenuContent>
            <Menu.Item value="unavailable">
              <Tooltip content="Item reason" open={isMenuOpen} placement="bottom">
                <span>Unavailable action</span>
              </Tooltip>
            </Menu.Item>
            <Menu.Item value="second">Second action</Menu.Item>
            <Menu.Item value="third">Third action</Menu.Item>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );

  return inPopover ? (
    <Popover.Root open positioning={POPOVER_AT}>
      <Portal>
        <Popover.Positioner>
          <PopoverContent>
            <div style={{ width: '120px' }}>{field}</div>
            {/* Room under the field, so its tip lies over the popover. */}
            <div style={{ height: '80px' }} />
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  ) : (
    <div style={{ left: '24px', position: 'absolute', top: '40px' }}>{field}</div>
  );
};

/** What the browser paints on top at a tip's centre; tips let the pointer through, so the probe lets this one hit. */
const paintedAtCentreOf = (text: string) => {
  const tip = [...document.querySelectorAll<HTMLElement>('[role="tooltip"]')].find((each) => each.textContent === text);
  if (!tip) {
    return null;
  }
  tip.style.pointerEvents = 'auto';
  const rect = tip.getBoundingClientRect();
  return document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
};

it.each([
  { where: 'on the page', inPopover: false, isMenuTriggerOutside: false },
  { where: 'in a popover', inPopover: true, isMenuTriggerOutside: false },
  { where: 'in a popover, inside its menu trigger', inPopover: true, isMenuTriggerOutside: true },
])("paints a tip $where beneath its menu, and a menu item's tip above the menu", async (props) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <MediaFieldWithMenu {...props} />
      </ChakraProvider>
    )
  );
  const field = document.getElementById(FIELD_ID)!;

  await userEvent.hover(field);
  await expect.poll(() => paintedAtCentreOf('Media name')?.textContent).toBe('Media name');

  // Near the corner, so the menu opens over the tip below the field.
  await userEvent.click(field, { button: 'right', position: { x: 4, y: 4 } });
  const menu = () => document.querySelector('[data-scope="menu"][data-part="content"]');

  await expect.poll(() => !!menu()?.contains(paintedAtCentreOf('Media name'))).toBe(true);
  await expect.poll(() => paintedAtCentreOf('Item reason')?.textContent).toBe('Item reason');
});
