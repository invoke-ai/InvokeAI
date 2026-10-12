import type * as LayoutPresetActivationModule from '@workbench/layoutPresetActivation';

import { ChakraProvider } from '@chakra-ui/react';
import { useDebouncedValue } from '@platform/react/useDebouncedValue';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { createWorkbenchStore, type WorkbenchInternalStore, type WorkbenchSnapshot } from '@workbench/workbenchStore';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

let store: WorkbenchInternalStore;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        'topbar.presets.layoutPreset': 'Layout preset',
        'topbar.presets.reorderInstructions':
          'To reorder a layout preset, press Space. Use the Left and Right arrow keys to move it, then press Space to drop or Escape to cancel.',
        'topbar.presets.saveAsTooltip': 'Save this layout as a new preset',
        'topbar.presets.unsaved': 'Unsaved changes',
        'topbar.presets.unsavedLayoutChanges': 'Unsaved layout changes',
      })[key] ?? key,
  }),
}));

// The real selection and debounce mechanics over a test-owned store; only the provider lookup is replaced.
vi.mock('@workbench/WorkbenchContext', () => {
  type Equality<Selected> = (left: Selected, right: Selected) => boolean;
  const useWorkbenchSelector = <Selected,>(
    selector: (snapshot: WorkbenchSnapshot) => Selected,
    isEqual: Equality<Selected> = shallowEqual
  ) => useExternalStoreSelector(store.subscribe, store.getSnapshot, selector, isEqual);

  return {
    useActiveProjectSelector: <Selected,>(
      selector: (project: WorkbenchSnapshot['activeProject']) => Selected,
      isEqual?: Equality<Selected>
    ) => useWorkbenchSelector((snapshot) => selector(snapshot.activeProject), isEqual),
    useDebouncedWorkbenchSelector: <Selected,>(
      selector: (snapshot: WorkbenchSnapshot) => Selected,
      debounceMs = 300,
      isEqual: Equality<Selected> = Object.is,
      settlesImmediately?: (previous: Selected, next: Selected) => boolean
    ) => useDebouncedValue(useWorkbenchSelector(selector, isEqual), debounceMs, { isEqual, settlesImmediately }),
    useWorkbenchCommands: () => store.commands,
    useWorkbenchSelector,
  };
});

vi.mock('@workbench/layoutPresetActivation', async (importOriginal) => {
  const actual = await importOriginal<typeof LayoutPresetActivationModule>();

  return {
    ...actual,
    loadLayoutPresetWidgets: () => Promise.resolve(),
    preloadLayoutPresetWidgets: () => undefined,
  };
});

vi.mock('./useTopbarShortcut', () => ({ useTopbarShortcut: () => null }));

import { LayoutPresetAdminDialogs } from './LayoutPresetAdminDialogs';
import { LayoutPresetStrip } from './LayoutPresetStrip';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderStrip = async ({ withAdminDialogs = false } = {}) => {
  host = document.createElement('div');
  host.style.width = '900px';
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <LayoutPresetStrip />
        {withAdminDialogs ? <LayoutPresetAdminDialogs /> : null}
      </ChakraProvider>
    );
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });
};

const presetTabIds = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[role="tab"][data-layout-preset-id]')).map(
    (element) => element.dataset.layoutPresetId
  );

const presetTab = (id: string): HTMLButtonElement | null =>
  document.querySelector<HTMLButtonElement>(`[role="tab"][data-layout-preset-id="${id}"]`);

const mouse = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new MouseEvent(type, { bubbles: true, button: 0, buttons: type === 'mouseup' ? 0 : 1, clientX, clientY })
  );
};

// A real press emits `pointerdown` before `mousedown`. dnd-kit's `MouseSensor`
// only listens to the latter, but the strip acknowledges on the former, so a
// drag simulation that skips it does not exercise what a user actually does.
const pressDown = (target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent('pointerdown', { bubbles: true, button: 0, cancelable: true, clientX, clientY })
  );
  mouse('mousedown', target, clientX, clientY);
};

const settle = (ms = 200): Promise<void> =>
  new Promise((resolve) => {
    globalThis.setTimeout(resolve, ms);
  });

const POLL = { timeout: 5000 } as const;

const openPresetMenuItem = async (presetId: string, item: string) => {
  const menuItem = () => document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${item}"]`);

  await act(() => userEvent.click(presetTab(presetId)!, { button: 'right' }));
  await expect.poll(menuItem, POLL).not.toBeNull();
  await act(() => userEvent.click(menuItem()!));
  await expect.poll(() => document.querySelector('[role="menu"]'), POLL).toBeNull();
};

const nextFrame = (): Promise<void> =>
  new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });

const touch = (
  type: string,
  target: EventTarget,
  touchTarget: EventTarget,
  clientX: number,
  clientY: number
): TouchEvent => {
  const point = new Touch({ clientX, clientY, identifier: 1, target: touchTarget });
  const isEnd = type === 'touchend' || type === 'touchcancel';
  const event = new TouchEvent(type, {
    bubbles: true,
    cancelable: true,
    changedTouches: [point],
    targetTouches: isEnd ? [] : [point],
    touches: isEnd ? [] : [point],
  });

  target.dispatchEvent(event);

  return event;
};

beforeEach(() => {
  store = createWorkbenchStore();
  store.commands.layout.createPreset('custom-1', 'Custom', 'star');
  // Saving as a new preset moves the project onto it; these tests start on Compose.
  store.commands.layout.applyPreset('compose');
  store.commands.layout.reorderPresets('custom-1', 'edit');
  store.commands.layout.renamePreset('compose', 'Writing');
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('LayoutPresetStrip', () => {
  it('renders the account order without preset descriptions and still activates a clicked tab', async () => {
    await renderStrip();

    expect(presetTabIds()).toEqual(['compose', 'custom-1', 'edit', 'video', 'automate']);
    expect(presetTab('compose')).toHaveAttribute('aria-label', 'Writing');
    expect(document.body.textContent).not.toContain('Text to image');

    await act(() => userEvent.click(presetTab('edit')!));

    expect(store.getSnapshot().activeProject.layout.presetId).toBe('edit');
  });

  // The menu's layer must be gone before an admin dialog mounts, or zag dismisses the dialog as nested above it.
  it.each(['save-as', 'edit'] as const)(
    'animates the %s dialog out instead of unmounting it on close',
    async (kind) => {
      await renderStrip({ withAdminDialogs: true });

      if (kind === 'save-as') {
        await act(() =>
          userEvent.click(document.querySelector<HTMLElement>('[aria-label="Save this layout as a new preset"]')!)
        );
      } else {
        await openPresetMenuItem('custom-1', 'edit-preset');
      }
      await expect.poll(() => document.querySelector('[role="dialog"]')?.getAttribute('data-state')).toBe('open');
      const dialog = document.querySelector('[role="dialog"]')!;

      const frames = await recordDialogExit(dialog, () => act(() => userEvent.keyboard('{Escape}')));

      // An unmounted dialog never reaches its closed state, so it cannot animate out; a retained one does, then leaves.
      expect(closingFrames(frames)).not.toHaveLength(0);
      await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();
    }
  );

  it('keeps the edit and delete dialogs opened from a preset menu', async () => {
    await renderStrip({ withAdminDialogs: true });

    await openPresetMenuItem('custom-1', 'edit-preset');

    const editDialog = document.querySelector<HTMLElement>('[role="dialog"]');
    expect(editDialog?.textContent).toMatch(/^topbar\.presets\.edit/);
    expect(document.querySelector('[role="menu"]')).toBeNull();

    const name = editDialog!.querySelector<HTMLInputElement>('input[name="layout-preset-name"]')!;
    await act(() => userEvent.clear(name));
    await act(() => userEvent.type(name, 'Renamed'));
    await act(() =>
      userEvent.click(
        Array.from(editDialog!.querySelectorAll('button')).find(
          (button) => button.textContent === 'topbar.presets.save'
        )!
      )
    );
    await act(() => settle());

    expect(presetTab('custom-1')).toHaveAttribute('aria-label', 'Renamed');

    await openPresetMenuItem('custom-1', 'delete-preset');

    const deleteDialog = document.querySelector<HTMLElement>('[role="alertdialog"], [role="dialog"]');
    expect(deleteDialog?.textContent).toMatch(/topbar\.presets\.deleteQuestion/);
    await act(() =>
      userEvent.click(
        Array.from(deleteDialog!.querySelectorAll('button')).find(
          (button) => button.textContent === 'topbar.presets.delete'
        )!
      )
    );
    await act(() => settle());

    expect(presetTabIds()).not.toContain('custom-1');
  });

  // Desktop tabs activate on press, including the press that begins reordering.
  it('selects a dragged tab on the press and reorders it in the same gesture', async () => {
    await renderStrip();
    await act(() => userEvent.click(presetTab('edit')!));
    expect(store.getSnapshot().activeProject.layout.presetId).toBe('edit');
    const source = presetTab('compose');
    const target = presetTab('automate');
    expect(source).not.toBeNull();
    expect(target).not.toBeNull();
    const start = source!.getBoundingClientRect();
    const end = target!.getBoundingClientRect();
    const startX = start.left + start.width / 2;
    const startY = start.top + start.height / 2;
    const endX = end.left + end.width / 2;

    await act(() => pressDown(source!, startX, startY));

    expect(presetTab('compose')).toHaveAttribute('aria-selected', 'true');
    expect(presetTab('edit')).toHaveAttribute('aria-selected', 'false');

    await act(() => mouse('mousemove', source!.ownerDocument, startX + 8, startY));
    await act(() => mouse('mousemove', source!.ownerDocument, endX, startY));
    await act(() => mouse('mouseup', source!.ownerDocument, endX, startY));
    await act(() => nextFrame());

    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['custom-1', 'edit', 'video', 'automate', 'compose']);
    expect(store.getSnapshot().activeProject.layout.presetId).toBe('compose');
  });

  it('supports keyboard reordering with sortable drag semantics', async () => {
    store.commands.layout.applyPreset('compose');
    await renderStrip();
    const source = presetTab('compose');
    expect(source).not.toBeNull();
    expect(source).toHaveAttribute('aria-roledescription', 'sortable');
    const instructions = document.getElementById(source!.getAttribute('aria-describedby') ?? '');
    expect(instructions?.textContent).toContain('press Space');
    expect(instructions?.textContent).not.toContain('Enter');
    expect(
      Array.from(document.querySelectorAll<HTMLElement>('[role="tab"]')).filter((tab) => tab.tabIndex === 0)
    ).toEqual([source]);
    source!.focus();

    await act(() => userEvent.keyboard('{Space}'));
    await act(() => userEvent.keyboard('{End}'));
    await act(() => userEvent.keyboard('{ArrowDown}'));
    expect(document.querySelector('[role="menu"]')).toBeNull();
    await act(() => userEvent.keyboard('{ArrowRight}'));
    await act(() => userEvent.keyboard('{Space}'));

    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['custom-1', 'compose', 'edit', 'video', 'automate']);
    expect(store.getSnapshot().activeProject.layout.presetId).toBe('compose');
    expect(store.getSnapshot().account.activeLayoutPresetId).toBe('compose');

    const movedSource = presetTab('compose');
    movedSource?.focus();
    await act(() => userEvent.keyboard('{Space}'));
    expect(movedSource?.style.opacity).toBe('0.5');
    await act(() => userEvent.keyboard('{Tab}'));
    expect(movedSource?.style.opacity).not.toBe('0.5');
  });

  it('keeps a non-overflowing strip stationary while dragging near its right edge', async () => {
    await renderStrip();
    const scrollContainer = document.querySelector<HTMLElement>('[data-layout-preset-scroll]');
    const source = presetTab('compose');
    expect(scrollContainer).not.toBeNull();
    expect(source).not.toBeNull();
    Object.defineProperty(scrollContainer!, 'scrollWidth', {
      configurable: true,
      value: scrollContainer!.clientWidth,
    });
    expect(scrollContainer!.scrollWidth).toBeLessThanOrEqual(scrollContainer!.clientWidth);
    const sourceRect = source!.getBoundingClientRect();
    const containerRect = scrollContainer!.getBoundingClientRect();
    const startX = sourceRect.left + sourceRect.width / 2;
    const startY = sourceRect.top + sourceRect.height / 2;
    const edgeX = containerRect.right - 2;

    await act(() => mouse('mousedown', source!, startX, startY));
    await act(() => mouse('mousemove', source!.ownerDocument, startX + 8, startY));
    await act(() => mouse('mousemove', source!.ownerDocument, edgeX, startY));
    await act(
      () =>
        new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 150);
        })
    );

    expect(scrollContainer!.scrollLeft).toBe(0);

    await act(() => mouse('mouseup', source!.ownerDocument, edgeX, startY));
  });

  it('keeps an overflowing strip stationary while dragging near its right edge', async () => {
    for (let index = 2; index <= 8; index += 1) {
      store.commands.layout.createPreset(`custom-${index}`, `Custom ${index}`, 'star');
    }
    await renderStrip();
    const scrollContainer = document.querySelector<HTMLElement>('[data-layout-preset-scroll]');
    const source = presetTab('compose');
    expect(scrollContainer).not.toBeNull();
    expect(source).not.toBeNull();
    expect(scrollContainer!.scrollWidth).toBeGreaterThan(scrollContainer!.clientWidth);
    const sourceRect = source!.getBoundingClientRect();
    const containerRect = scrollContainer!.getBoundingClientRect();
    const startX = sourceRect.left + sourceRect.width / 2;
    const startY = sourceRect.top + sourceRect.height / 2;
    const edgeX = containerRect.right - 2;

    await act(() => mouse('mousedown', source!, startX, startY));
    await act(() => mouse('mousemove', source!.ownerDocument, startX + 8, startY));
    await act(() => mouse('mousemove', source!.ownerDocument, edgeX, startY));
    await act(
      () =>
        new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 250);
        })
    );

    expect(scrollContainer!.scrollLeft).toBe(0);

    await act(() => mouse('mouseup', source!.ownerDocument, edgeX, startY));
    const scrollLeftAfterDrop = scrollContainer!.scrollLeft;
    await act(
      () =>
        new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 100);
        })
    );

    expect(scrollLeftAfterDrop).toBe(0);
    expect(scrollContainer!.scrollLeft).toBe(0);
  });

  it('preserves horizontal touch panning and reorders after a long press', async () => {
    await renderStrip();
    const source = presetTab('compose');
    const target = presetTab('automate');
    expect(source).not.toBeNull();
    expect(target).not.toBeNull();
    expect(getComputedStyle(source!).touchAction).toBe('pan-x');
    const start = source!.getBoundingClientRect();
    const end = target!.getBoundingClientRect();
    const startX = start.left + start.width / 2;
    const startY = start.top + start.height / 2;
    const endX = end.left + end.width / 2;

    await act(() => touch('touchstart', source!, source!, startX, startY));
    expect(source!.style.opacity).not.toBe('0.5');
    await act(
      () =>
        new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 100);
        })
    );
    expect(source!.style.opacity).not.toBe('0.5');
    await act(
      () =>
        new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 200);
        })
    );
    expect(source!.style.opacity).toBe('0.5');
    await act(() => touch('touchmove', source!, source!, endX, startY));
    await act(() => touch('touchend', source!, source!, endX, startY));

    expect(source!.style.opacity).not.toBe('0.5');
    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['custom-1', 'edit', 'video', 'automate', 'compose']);
  });

  it('leaves a short horizontal touch gesture to an overflowing scroll strip', async () => {
    for (let index = 2; index <= 8; index += 1) {
      store.commands.layout.createPreset(`custom-${index}`, `Custom ${index}`, 'star');
    }
    await renderStrip();
    const scrollContainer = document.querySelector<HTMLElement>('[data-layout-preset-scroll]');
    const source = presetTab('compose');
    expect(scrollContainer).not.toBeNull();
    expect(source).not.toBeNull();
    expect(scrollContainer!.scrollWidth).toBeGreaterThan(scrollContainer!.clientWidth);
    const orderBeforePan = store.getSnapshot().account.layoutPresetOrder;
    const start = source!.getBoundingClientRect();
    const startX = start.left + start.width / 2;
    const startY = start.top + start.height / 2;

    await act(() => touch('touchstart', source!, source!, startX, startY));
    const move = touch('touchmove', source!, source!, startX - 40, startY);
    scrollContainer!.scrollLeft = 40;
    await act(() => touch('touchend', source!, source!, startX - 40, startY));

    expect(move.defaultPrevented).toBe(false);
    expect(scrollContainer!.scrollLeft).toBe(40);
    expect(source!.style.opacity).not.toBe('0.5');
    expect(store.getSnapshot().account.layoutPresetOrder).toBe(orderBeforePan);
  });

  describe('unsaved arrangements', () => {
    const UNSAVED = 'Unsaved layout changes';
    const openTooltip = () =>
      [...document.querySelectorAll<HTMLElement>('[data-scope="tooltip"][data-part="content"]')].find(
        (element) => element.dataset.state === 'open'
      ) ?? null;
    const unsavedDot = (id: string) => presetTab(id)?.querySelector('[data-unsaved-dot]') ?? null;
    const label = (id: string) => presetTab(id)?.getAttribute('aria-label') ?? '';
    const activePresetId = () => store.getSnapshot().activeProject.layout.presetId;
    const waitForActivePreset = (id: string) => expect.poll(activePresetId, POLL).toBe(id);

    /** Rearrange `presetId` in the store, leaving it active. */
    const rearrange = (presetId: string, sizePx: number) => {
      store.commands.layout.applyPreset(presetId);
      store.commands.layout.setRegionSize('right', sizePx);
    };

    it('marks every preset with unsaved changes over its icon and in its accessible name', async () => {
      await renderStrip();
      expect(unsavedDot('compose')).toBeNull();
      expect(label('compose')).toBe('Writing');

      await act(() => store.commands.layout.setRegionSize('right', 401));

      // Within one preset the dot follows the settled arrangement.
      await expect.poll(() => label('compose'), POLL).toBe(`Writing, ${UNSAVED}`);
      // On the icon, not after the label: the dot sits inside the icon's box.
      expect(unsavedDot('compose')?.parentElement?.querySelector('svg')).not.toBeNull();

      await act(() => userEvent.click(presetTab('edit')!));
      await waitForActivePreset('edit');

      // Leaving Compose keeps its arrangement as unsaved; Edit opened clean, at once rather than after the settle.
      expect(label('compose')).toBe(`Writing, ${UNSAVED}`);
      expect(label('edit')).not.toContain(UNSAVED);
      expect(unsavedDot('edit')).toBeNull();

      await act(() => store.commands.layout.setRegionSize('right', 433));

      await expect.poll(() => unsavedDot('edit'), POLL).not.toBeNull();
      expect(unsavedDot('compose')).not.toBeNull();
      expect(unsavedDot('video')).toBeNull();
    });

    it('shows the arriving preset’s own state at once, whichever way the switch goes', async () => {
      rearrange('video', 401);
      rearrange('edit', 433);
      await renderStrip();
      await expect.poll(() => label('edit'), POLL).toContain(UNSAVED);

      // Unsaved → clean: no dot borrowed from the preset just left, in the first frames after the switch.
      await act(() => userEvent.click(presetTab('automate')!));
      await waitForActivePreset('automate');
      expect(unsavedDot('automate')).toBeNull();
      expect(label('automate')).not.toContain(UNSAVED);
      expect(unsavedDot('edit')).not.toBeNull();

      // Clean → unsaved: the arriving preset's dot and name are there at once, not after the settle.
      await act(() => userEvent.click(presetTab('video')!));
      await waitForActivePreset('video');
      expect(unsavedDot('video')).not.toBeNull();
      expect(label('video')).toContain(UNSAVED);
      expect(unsavedDot('automate')).toBeNull();
    });

    it('explains the dot on keyboard focus whichever tab focus came from, and stays quiet on a clean preset', async () => {
      rearrange('video', 401);
      rearrange('compose', 433);
      await renderStrip();

      // A clean tab under keyboard focus: focus opens a tip at once when there is one, so none here is not vacuous.
      presetTab('compose')!.focus();
      await act(() => userEvent.keyboard('{ArrowRight}'));
      await act(() => userEvent.keyboard('{ArrowRight}'));
      await waitForActivePreset('edit');
      expect(document.activeElement).toBe(presetTab('edit'));
      expect(openTooltip()).toBeNull();

      // Clean → unsaved by keyboard.
      await act(() => userEvent.keyboard('{ArrowRight}'));
      await waitForActivePreset('video');
      expect(document.activeElement).toBe(presetTab('video'));
      await expect.poll(() => openTooltip()?.textContent, POLL).toBe(UNSAVED);
      expect(label('video')).toContain(UNSAVED);
    });

    it('hands the tip from one unsaved preset to the next as the arrow keys move between them', async () => {
      rearrange('automate', 461);
      rearrange('video', 401);
      await renderStrip();

      // Keyboard focus lands on the selected (unsaved) tab and its tip shows.
      await act(() => userEvent.tab());
      expect(document.activeElement).toBe(presetTab('video'));
      await expect.poll(() => openTooltip()?.textContent, POLL).toBe(UNSAVED);

      await act(() => userEvent.keyboard('{ArrowRight}'));
      await waitForActivePreset('automate');
      expect(document.activeElement).toBe(presetTab('automate'));
      expect(label('automate')).toContain(UNSAVED);
      await expect.poll(() => openTooltip()?.textContent, POLL).toBe(UNSAVED);
      // Still open once the switch has settled, not closed by it.
      await act(() => settle(400));
      expect(openTooltip()?.textContent).toBe(UNSAVED);
    });

    describe('never opens the tip on a tab nobody is on', () => {
      // The tip opens on focus without a delay and on hover after 400 ms; these waits are well past both.
      const QUIET_MS = 700;
      const unsavedTip = () => (openTooltip()?.textContent === UNSAVED ? openTooltip() : null);
      const dirtyCompose = async (sizePx: number) => {
        await act(() => store.commands.layout.setRegionSize('right', sizePx));
        await expect.poll(() => label('compose'), POLL).toContain(UNSAVED);
      };

      it('after keyboard focus left a clean tab that then became unsaved', async () => {
        await renderStrip();
        await act(() => userEvent.tab());
        expect(document.activeElement).toBe(presetTab('compose'));
        await act(() => (document.activeElement as HTMLElement).blur());

        await dirtyCompose(401);
        await act(() => settle(QUIET_MS));
        expect(unsavedTip()).toBeNull();

        // The tip itself works: focus brings it.
        await act(() => presetTab('compose')!.focus());
        await expect.poll(unsavedTip, POLL).not.toBeNull();
      });

      it('after a focused unsaved tab was reverted in place, left, and changed again', async () => {
        await renderStrip();
        await dirtyCompose(401);
        await act(() => userEvent.tab());
        await expect.poll(unsavedTip, POLL).not.toBeNull();

        await act(() => store.commands.layout.reset());
        await expect.poll(() => label('compose'), POLL).not.toContain(UNSAVED);
        await expect.poll(unsavedTip, POLL).toBeNull();
        await act(() => (document.activeElement as HTMLElement).blur());

        await dirtyCompose(433);
        await act(() => settle(QUIET_MS));
        expect(unsavedTip()).toBeNull();
      });

      it('after the pointer left a clean tab that then became unsaved', async () => {
        await renderStrip();
        await act(() => userEvent.hover(presetTab('compose')!));
        await act(() => settle(QUIET_MS));
        expect(unsavedTip()).toBeNull();
        await act(() => userEvent.unhover(presetTab('compose')!));

        await dirtyCompose(401);
        await act(() => settle(QUIET_MS));
        expect(unsavedTip()).toBeNull();

        await act(() => userEvent.hover(presetTab('compose')!));
        await expect.poll(unsavedTip, POLL).not.toBeNull();
      });
    });

    it('explains the dot on hover', async () => {
      rearrange('compose', 401);
      store.commands.layout.applyPreset('edit');
      await renderStrip();

      await act(() => userEvent.hover(presetTab('compose')!));
      await expect.poll(() => openTooltip()?.textContent, POLL).toBe(UNSAVED);
      await act(() => userEvent.unhover(presetTab('compose')!));
      await expect.poll(openTooltip, POLL).toBeNull();
    });

    it('saves or reverts an inactive preset’s unsaved arrangement from its menu without switching', async () => {
      rearrange('compose', 401);
      store.commands.layout.applyPreset('edit');
      await renderStrip();

      await openPresetMenuItem('compose', 'revert-layout');

      await expect.poll(() => unsavedDot('compose'), POLL).toBeNull();
      expect(activePresetId()).toBe('edit');

      rearrange('compose', 461);
      await act(() => store.commands.layout.applyPreset('edit'));
      await openPresetMenuItem('compose', 'save-layout');

      await expect.poll(() => unsavedDot('compose'), POLL).toBeNull();
      expect(activePresetId()).toBe('edit');
      expect(store.getSnapshot().account.layoutPresetOverrides?.compose?.widgetRegions.right.sizePx).toBe(461);
    });

    it('stays on an unsaved preset when it is pressed again while a slow switch away is still loading', async () => {
      store = createWorkbenchStore(store.getState(), {
        isLoaded: () => false,
        // Never finishes: the switch waits out its bounded deadline, the window in which a second press lands.
        loadLayoutPresetWidgets: () => new Promise(() => {}),
      });
      await act(() => store.commands.layout.setRegionSize('right', 401));
      await renderStrip();
      await expect.poll(() => label('compose'), POLL).toContain(UNSAVED);

      await act(() => userEvent.click(presetTab('edit')!));
      expect(presetTab('edit')).toHaveAttribute('aria-selected', 'true');
      await act(() => userEvent.click(presetTab('compose')!));

      // Well past the activation deadline: nothing switched, and Compose kept its unsaved arrangement.
      await act(() => settle(600));
      expect(activePresetId()).toBe('compose');
      expect(store.getSnapshot().activeProject.widgetRegions.right.sizePx).toBe(401);
      expect(presetTab('compose')).toHaveAttribute('aria-selected', 'true');
      expect(label('compose')).toContain(UNSAVED);
    });
  });
});
