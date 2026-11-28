import type * as LayoutPresetActivationModule from '@workbench/layoutPresetActivation';

import { ChakraProvider } from '@chakra-ui/react';
import { dragWithMouse } from '@platform/browser/mouseDrag.testing';
import { useDebouncedValue } from '@platform/react/useDebouncedValue';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { system } from '@theme/system';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { createWorkbenchStore, type WorkbenchInternalStore, type WorkbenchSnapshot } from '@workbench/workbenchStore';
import { rearrangeWithoutCustomizeInEachTest } from '@workbench/workspace/structureLock.testing';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

let store: WorkbenchInternalStore;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { count?: number }) =>
      ({
        'topbar.presets.moreAvailable': `More layout presets, ${options?.count} available`,
        'topbar.presets.builtInLabels.automate': 'Arbeitsablauf',
        'topbar.presets.builtInLabels.edit': 'Edit',
        'topbar.presets.builtInDescriptions.automate': 'Build and run workflows',
        'topbar.presets.builtInDescriptions.compose': 'Generate images from text',
        'topbar.presets.builtInDescriptions.edit': 'Edit and refine images',
        'topbar.presets.builtInDescriptions.video': 'Generate video',
        'topbar.presets.newWorkspace': 'New workspace',
        'topbar.presets.reorderInstructions':
          'To reorder a layout preset, press Space. Use the Left and Right arrow keys to move it, then press Space to drop or Escape to cancel.',
        'topbar.presets.workspaces': 'Workspaces',
      })[key] ?? key,
  }),
}));

vi.mock('@workbench/WorkbenchContext', () => {
  type Equality<Selected> = (left: Selected, right: Selected) => boolean;
  const useWorkbenchSelector = <Selected,>(
    selector: (snapshot: WorkbenchSnapshot) => Selected,
    isEqual: Equality<Selected> = shallowEqual
  ) => useExternalStoreSelector(store.subscribe, store.getSnapshot, selector, isEqual);

  return {
    useActiveWorkspaceId: () => useWorkbenchSelector((snapshot) => snapshot.activeProject.workspaceId),
    useActiveLayoutSelector: <Selected,>(
      selector: (layout: WorkbenchSnapshot['activeLayout']) => Selected,
      isEqual?: Equality<Selected>
    ) => useWorkbenchSelector((snapshot) => selector(snapshot.activeLayout), isEqual),
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
    useOptionalWorkbenchSelector: <Selected,>(
      selector: (snapshot: WorkbenchSnapshot) => Selected,
      _fallback: Selected,
      isEqual?: Equality<Selected>
    ) => useWorkbenchSelector(selector, isEqual),
    useWorkbenchCommands: () => store.commands,
    useWorkbenchQueries: () => store.queries,
    useWorkbenchSelector,
  };
});

vi.mock('@workbench/layoutPresetActivation', async (importOriginal) => {
  const actual = await importOriginal<typeof LayoutPresetActivationModule>();

  return {
    ...actual,
    loadLayoutWidgets: () => Promise.resolve(),
    preloadLayoutWidgets: () => undefined,
  };
});

const COMPOSE_SHORTCUT = { aria: 'Alt+1', display: 'Alt+1', parts: ['alt', '1'] };
vi.mock('@workbench/hotkeys/useShortcutBinding', () => ({
  useShortcutBinding: (commandId: string) => (commandId === 'app.selectComposePreset' ? COMPOSE_SHORTCUT : null),
  useShortcutLabel: () => null,
}));

import { LayoutPresetAdminDialogs } from './LayoutPresetAdminDialogs';
import { LayoutPresetManagerDialog } from './LayoutPresetManagerDialog';
import { LayoutPresetStrip } from './LayoutPresetStrip';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderStrip = async ({ withAdminDialogs = false, withManager = false } = {}) => {
  host = document.createElement('div');
  host.style.width = '900px';
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <LayoutPresetStrip />
        {withAdminDialogs ? <LayoutPresetAdminDialogs /> : null}
        {withManager ? <LayoutPresetManagerDialog /> : null}
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

// A real press emits `pointerdown` before `mousedown`. dnd-kit's `MouseSensor` only listens to the latter, but the
// strip acknowledges on the former, so a drag simulation that skips it does not exercise what a user actually does.
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

const openTooltip = () =>
  [...document.querySelectorAll<HTMLElement>('[data-scope="tooltip"][data-part="content"]')].find(
    (element) => element.dataset.state === 'open'
  ) ?? null;
const NARROW_VIEWPORT = { height: 900, width: 1024 } as const;

const runAtViewport = (size: { height: number; width: number }) => {
  let original = { height: 0, width: 0 };

  beforeEach(async () => {
    original = { height: globalThis.innerHeight, width: globalThis.innerWidth };
    await page.viewport(size.width, size.height);
  });
  afterEach(() => page.viewport(original.width, original.height));
};

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

// Reordering from the strip is structure: it works here for an account that rearranges without Customize. `locked
// workspaces` below covers the default.
rearrangeWithoutCustomizeInEachTest();

beforeEach(() => {
  store = createWorkbenchStore();
  store.commands.layout.beginCustomize({ defaultRoute: null, iconId: 'star', id: 'custom-1', label: 'Custom' });
  store.commands.layout.finishCustomize();
  store.commands.layout.applyPreset('compose');
  store.commands.layout.reorderPresets('custom-1', 'edit');
  store.commands.layout.applyPreset('compose');
  store.commands.layout.beginCustomize();
  store.commands.layout.setCustomizeDraft({ label: 'Writing' });
  store.commands.layout.finishCustomize();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('the active preset’s tab', () => {
  const centerFront = () => store.getSnapshot().activeLayout.widgetRegions.center.activeInstanceId;
  const leaveHome = () =>
    act(() =>
      store.commands.widgets.select({
        projectId: store.getSnapshot().activeProject.id,
        region: 'center',
        widgetId: 'gallery:center',
      })
    );

  it('goes home when clicked or activated with Enter', async () => {
    await renderStrip();
    await leaveHome();

    await act(() => userEvent.click(presetTab('compose')!));
    expect(centerFront()).toBe('preview');

    await leaveHome();
    presetTab('compose')!.focus();
    await act(() => userEvent.keyboard('{Enter}'));
    expect(centerFront()).toBe('preview');
  });

  it('never goes home when it is dragged', async () => {
    await renderStrip();
    await leaveHome();
    const source = presetTab('compose')!;
    const { left, top, width, height } = source.getBoundingClientRect();

    // A real drag: the browser clicks the tab it was pressed and released on.
    source.dispatchEvent(
      new PointerEvent('pointerdown', {
        bubbles: true,
        button: 0,
        clientX: left + width / 2,
        clientY: top + height / 2,
      })
    );
    await dragWithMouse(source, presetTab('automate')!);
    await act(() => presetTab('compose')!.click());
    await act(() => nextFrame());
    expect(store.getSnapshot().account.layoutPresetOrder?.at(-1)).toBe('compose');
    expect(centerFront()).toBe('gallery:center');
  });

  it('never goes home when reordered with the keyboard', async () => {
    await renderStrip();
    await leaveHome();

    presetTab('compose')!.focus();
    await act(() => userEvent.keyboard('{Space}'));
    await act(() => userEvent.keyboard('{ArrowRight}'));
    await act(() => userEvent.keyboard('{Space}'));
    expect(store.getSnapshot().account.layoutPresetOrder?.[0]).not.toBe('compose');
    expect(centerFront()).toBe('gallery:center');
  });
});

describe('LayoutPresetStrip', () => {
  it('stays on the workspace when its tab is pressed again while a slow switch away is still loading', async () => {
    store = createWorkbenchStore(store.getState(), {
      isLoaded: () => false,
      loadLayoutWidgets: () => new Promise(() => {}),
    });
    await act(() => store.commands.layout.setRegionSize('right', 401));
    await renderStrip();

    await act(() => userEvent.click(presetTab('edit')!));
    expect(presetTab('edit')).toHaveAttribute('aria-selected', 'true');
    await act(() => userEvent.click(presetTab('compose')!));

    await act(() => settle(600));
    expect(store.getSnapshot().activeProject.workspaceId).toBe('compose');
    expect(store.getSnapshot().activeLayout.widgetRegions.right.sizePx).toBe(401);
    expect(presetTab('compose')).toHaveAttribute('aria-selected', 'true');
  });

  it('renders the account order without preset descriptions and still activates a clicked tab', async () => {
    await renderStrip();

    expect(presetTabIds()).toEqual(['compose', 'custom-1', 'edit', 'video', 'automate']);
    expect(presetTab('compose')).toHaveAttribute('aria-label', 'Writing');
    expect(document.body.textContent).not.toContain('Text to image');

    await act(() => userEvent.click(presetTab('edit')!));

    expect(store.getSnapshot().activeProject.workspaceId).toBe('edit');
  });

  it('keeps the delete confirmation opened from a workspace menu', async () => {
    await renderStrip({ withAdminDialogs: true });

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

  it('shows a built-in workspace by its translated name', async () => {
    await renderStrip();

    expect(presetTab('automate')).toHaveAttribute('aria-label', 'Arbeitsablauf');
  });

  it('selects a dragged tab on the press and reorders it in the same gesture', async () => {
    await renderStrip();
    await act(() => userEvent.click(presetTab('edit')!));
    expect(store.getSnapshot().activeProject.workspaceId).toBe('edit');
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
    expect(store.getSnapshot().activeProject.workspaceId).toBe('compose');
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
    expect(store.getSnapshot().activeProject.workspaceId).toBe('compose');
    expect(store.getSnapshot().account.activeLayoutPresetId).toBe('compose');

    const movedSource = presetTab('compose');
    movedSource?.focus();
    await act(() => userEvent.keyboard('{Space}'));
    expect(movedSource?.style.opacity).toBe('0.5');
    await act(() => userEvent.keyboard('{Tab}'));
    expect(movedSource?.style.opacity).not.toBe('0.5');
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

  describe('preset tips', () => {
    const tipText = () => openTooltip()?.textContent ?? null;
    const tipKeycap = () => openTooltip()?.querySelector('kbd') ?? null;

    describe('with the labels hidden', () => {
      runAtViewport(NARROW_VIEWPORT);

      it('names the preset under the pointer, with what a built-in is for and its shortcut', async () => {
        await renderStrip();

        await act(() => userEvent.hover(presetTab('compose')!));
        await expect.poll(tipText, POLL).toContain('Writing');
        expect(tipText()).not.toContain('Generate images from text');
        expect(tipKeycap()?.textContent).toContain('1');
        expect(presetTab('compose')).toHaveAttribute('aria-label', 'Writing');

        await act(() => userEvent.hover(presetTab('edit')!));
        await expect.poll(tipText, POLL).toBe('EditEdit and refine images');
        expect(tipKeycap()).toBeNull();

        await act(() => userEvent.hover(presetTab('custom-1')!));
        await expect.poll(tipText, POLL).toBe('Custom');
      });
    });
  });
});

describe('locked workspaces', () => {
  beforeEach(() => {
    void patchWorkbenchPreferences({ rearrangeWithoutCustomize: false });
  });

  it('opens Manage workspaces from any tab’s menu and gives focus back to that tab', async () => {
    await renderStrip({ withManager: true });

    await act(() => userEvent.click(presetTab('edit')!, { button: 'right' }));
    await expect
      .poll(() => document.querySelector('[role="menuitem"][data-value="manage-presets"]'), POLL)
      .not.toBeNull();
    await act(() =>
      userEvent.click(document.querySelector<HTMLElement>('[role="menuitem"][data-value="manage-presets"]')!)
    );
    await expect.poll(() => document.querySelector('[role="dialog"]')?.getAttribute('data-state'), POLL).toBe('open');

    await act(() => userEvent.keyboard('{Escape}'));

    await expect.poll(() => document.querySelector('[role="dialog"]'), POLL).toBeNull();
    await expect.poll(() => document.activeElement, POLL).toBe(presetTab('edit'));
  });

  it('keeps the tab order when a tab is dragged, and does not announce tabs as sortable', async () => {
    await renderStrip();
    const order = presetTabIds();

    expect(presetTab('compose')).not.toHaveAttribute('aria-roledescription');
    await dragWithMouse(presetTab('compose')!, presetTab('automate')!);
    await act(() => nextFrame());

    expect(presetTabIds()).toEqual(order);
    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['compose', 'custom-1', 'edit', 'video', 'automate']);
  });

  it('offers Customize, Reset view and Manage workspaces in the active tab’s menu, and no Delete', async () => {
    const menuItems = () =>
      Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).map((item) => item.dataset.value);
    const menuItem = (value: string) => document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`);

    await act(() => store.commands.layout.setRegionSize('right', 401));
    await renderStrip();
    await act(() => settle(400));

    expect(presetTab('compose')).toHaveAttribute('aria-label', 'Writing');

    await act(() => userEvent.click(presetTab('compose')!, { button: 'right' }));
    await expect.poll(() => menuItem('customize'), POLL).not.toBeNull();
    expect(menuItems()).toEqual(['customize', 'reset-view', 'manage-presets']);

    await act(() => userEvent.click(menuItem('reset-view')!));

    expect(store.getSnapshot().activeProject.views.compose).toBeUndefined();
    expect(store.getSnapshot().activeLayout.widgetRegions.right.sizePx).toBe(450);

    await act(() => userEvent.click(presetTab('edit')!, { button: 'right' }));
    await expect.poll(() => menuItem('reset-view'), POLL).not.toBeNull();
    // Nothing to reset: the item stays, unavailable, and says why.
    expect(menuItem('reset-view')).toHaveAttribute('aria-disabled', 'true');
    await act(() => userEvent.keyboard('{Escape}'));

    await act(() => userEvent.click(presetTab('compose')!, { button: 'right' }));
    await expect.poll(() => menuItem('customize'), POLL).not.toBeNull();
    await act(() => userEvent.click(menuItem('customize')!));

    expect(store.getSnapshot().customizing?.[store.getSnapshot().activeProject.id]?.presetId).toBe('compose');
  });
});
