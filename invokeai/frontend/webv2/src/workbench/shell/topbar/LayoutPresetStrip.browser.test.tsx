import type * as LayoutPresetActivationModule from '@workbench/layoutPresetActivation';

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
    t: (key: string) =>
      ({
        'topbar.presets.layoutPreset': 'Layout preset',
        'topbar.presets.reorderInstructions':
          'To reorder a layout preset, press Space. Use the Left and Right arrow keys to move it, then press Space to drop or Escape to cancel.',
        'topbar.presets.saveAsTooltip': 'Save this layout as a new preset',
        'topbar.presets.unsaved': 'Unsaved changes',
      })[key] ?? key,
  }),
}));

vi.mock('@workbench/WorkbenchContext', () => {
  const useSnapshot = () => useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  return {
    useActiveProjectSelector: <Selected,>(
      selector: (project: ReturnType<typeof useSnapshot>['activeProject']) => Selected
    ) => selector(useSnapshot().activeProject),
    useDebouncedWorkbenchSelector: <Selected,>(selector: (snapshot: ReturnType<typeof useSnapshot>) => Selected) =>
      selector(useSnapshot()),
    useWorkbenchCommands: () => store.commands,
    useWorkbenchSelector: <Selected,>(selector: (snapshot: ReturnType<typeof useSnapshot>) => Selected) =>
      selector(useSnapshot()),
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

import { LayoutPresetStrip } from './LayoutPresetStrip';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderStrip = async () => {
  host = document.createElement('div');
  host.style.width = '900px';
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <LayoutPresetStrip />
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

    expect(presetTabIds()).toEqual(['compose', 'custom-1', 'edit', 'automate']);
    expect(presetTab('compose')).toHaveAttribute('aria-label', 'Writing');
    expect(document.body.textContent).not.toContain('Text to image');

    await act(() => userEvent.click(presetTab('edit')!));

    expect(store.getSnapshot().activeProject.layout.presetId).toBe('edit');
  });

  it('reorders a dragged tab without activating it', async () => {
    await renderStrip();
    await act(() => userEvent.click(presetTab('edit')!));
    const activeBeforeDrag = store.getSnapshot().activeProject.layout.presetId;
    const source = presetTab('compose');
    const target = presetTab('automate');
    expect(source).not.toBeNull();
    expect(target).not.toBeNull();
    const start = source!.getBoundingClientRect();
    const end = target!.getBoundingClientRect();
    const startX = start.left + start.width / 2;
    const startY = start.top + start.height / 2;
    const endX = end.left + end.width / 2;

    await act(() => mouse('mousedown', source!, startX, startY));
    await act(() => mouse('mousemove', source!.ownerDocument, startX + 8, startY));
    await act(() => mouse('mousemove', source!.ownerDocument, endX, startY));
    await act(() => mouse('mouseup', source!.ownerDocument, endX, startY));

    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['custom-1', 'edit', 'automate', 'compose']);
    expect(store.getSnapshot().activeProject.layout.presetId).toBe(activeBeforeDrag);
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

    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['custom-1', 'compose', 'edit', 'automate']);
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
    expect(store.getSnapshot().account.layoutPresetOrder).toEqual(['custom-1', 'edit', 'automate', 'compose']);
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
});
