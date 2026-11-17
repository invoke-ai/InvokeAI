import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { SegmentTabs, type SegmentTab } from './SegmentTabs';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const TABS = [
  { id: 'properties', label: 'Properties' },
  { id: 'transform', label: 'Transform' },
  { id: 'overview', label: 'Overview' },
] as const;
const noop = (): void => {};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** Computed opacity of each divider between tabs, left to right, once transitions settle. */
const dividerOpacities = async (showActivePanel: boolean, hoveredTab?: string): Promise<number[]> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <SegmentTabs
          activeId="properties"
          ariaLabel="Editor panes"
          idBase="panes"
          showActivePanel={showActivePanel}
          tabs={TABS}
          onSelect={noop}
        />
      </ChakraProvider>
    )
  );
  if (hoveredTab) {
    const tab = [...host.querySelectorAll<HTMLElement>('[role="tab"]')].find((el) => el.textContent === hoveredTab);
    await userEvent.hover(tab!);
  }
  const dividers = [...host.querySelectorAll<HTMLElement>('[role="tablist"] > [aria-hidden]')];
  await Promise.all(dividers.flatMap((divider) => divider.getAnimations().map((animation) => animation.finished)));
  return dividers.map((divider) => Number(getComputedStyle(divider).opacity));
};

it('hides only the dividers beside the shown tab', async () => {
  expect(await dividerOpacities(true)).toEqual([0, 1]);
});

it('keeps every divider when the selected tab has no shown panel', async () => {
  expect(await dividerOpacities(false)).toEqual([1, 1]);
});

it('hides the dividers beside a hovered tab as well as the shown one', async () => {
  expect(await dividerOpacities(true, 'Overview')).toEqual([0, 0]);
});

it('hides the dividers beside a hovered tab in a collapsed strip', async () => {
  expect(await dividerOpacities(false, 'Transform')).toEqual([0, 0]);
});

const renderTabs = (tabs: readonly SegmentTab[], activeId: string, onSelect: (id: string) => void = noop) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <SegmentTabs activeId={activeId} ariaLabel="Details" idBase="details" tabs={tabs} onSelect={onSelect} />
      </ChakraProvider>
    )
  );
  return (label: string) =>
    [...host!.querySelectorAll<HTMLElement>('[role="tab"]')].find((el) => el.textContent === label)!;
};

const WITH_DISABLED: readonly SegmentTab[] = [
  { id: 'details', label: 'Details' },
  { id: 'metadata', label: 'Metadata', disabled: true },
  { id: 'workflow', label: 'Workflow' },
  { id: 'graph', label: 'Graph', disabled: true },
];

it('marks a disabled tab and ignores clicks on it', async () => {
  const onSelect = vi.fn();
  const tab = renderTabs(WITH_DISABLED, 'details', onSelect);

  expect(tab('Metadata').getAttribute('aria-disabled')).toBe('true');
  expect(tab('Workflow').hasAttribute('aria-disabled')).toBe(false);
  expect(getComputedStyle(tab('Metadata')).opacity).toBe('0.5');

  // Playwright refuses to click an aria-disabled element; a pointer still can.
  act(() => tab('Metadata').click());
  expect(onSelect).not.toHaveBeenCalled();
  await userEvent.click(tab('Workflow'));
  expect(onSelect).toHaveBeenCalledExactlyOnceWith('workflow');
});

it('skips disabled tabs with the arrows, Home and End', async () => {
  const tab = renderTabs(WITH_DISABLED, 'details');
  const press = async (key: string) => {
    await act(() => userEvent.keyboard(`{${key}}`));
    return document.activeElement?.textContent;
  };
  act(() => tab('Details').focus());

  expect(await press('ArrowRight')).toBe('Workflow');
  // Past the disabled Graph, wrapping round to the start.
  expect(await press('ArrowRight')).toBe('Details');
  expect(await press('ArrowLeft')).toBe('Workflow');
  expect(await press('End')).toBe('Workflow');
  expect(await press('Home')).toBe('Details');
});

it('steps from a selected disabled tab to its enabled neighbours', async () => {
  const tab = renderTabs(WITH_DISABLED, 'graph');

  expect(tab('Graph').tabIndex).toBe(0);
  act(() => tab('Graph').focus());
  await act(() => userEvent.keyboard('{ArrowLeft}'));
  expect(document.activeElement).toBe(tab('Workflow'));
});

it('gives a hovered disabled tab no hover look, and keeps its dividers', async () => {
  const tab = renderTabs(WITH_DISABLED, 'details');
  const disabled = tab('Metadata');
  const look = () => {
    const style = getComputedStyle(disabled);
    return { background: style.backgroundColor, color: style.color };
  };
  const dividers = () =>
    [...host!.querySelectorAll<HTMLElement>('[role="tablist"] > [aria-hidden]')].map((divider) =>
      Number(getComputedStyle(divider).opacity)
    );
  const before = { dividers: dividers(), look: look() };

  await userEvent.hover(disabled, { force: true });
  await Promise.all(
    [disabled, ...host!.querySelectorAll<HTMLElement>('[aria-hidden]')].flatMap((element) =>
      element.getAnimations().map((animation) => animation.finished)
    )
  );

  expect(disabled.matches(':hover')).toBe(true);
  expect(look()).toEqual(before.look);
  expect(dividers()).toEqual(before.dividers);
});

// Labels as long as a German queue strip's, so five tabs cannot share a narrow rail.
const LONG_TABS: readonly SegmentTab[] = [
  { id: 'all', label: 'Alle 12.345' },
  { id: 'active', label: 'Aktiv 12.345' },
  { id: 'done', label: 'Erledigt 12.345' },
  { id: 'failed', label: 'Fehlgeschlagen 12.345' },
  { id: 'canceled', label: 'Abgebrochen 12.345' },
];
// How far the end fades reach into a scrolling strip; a revealed tab must sit clear of them.
const FADE_PX = 16;

const nextFrame = () =>
  new Promise((resolve) => {
    requestAnimationFrame(resolve);
  });

const renderScrollingTabs = (width: number, activeId = 'all', onSelect: (id: string) => void = noop) => {
  host = document.createElement('div');
  host.style.width = `${width}px`;
  document.body.append(host);
  root = createRoot(host);
  const render = (active: string) =>
    act(() =>
      root!.render(
        <ChakraProvider value={system}>
          <SegmentTabs
            isCompact
            isScrollable
            activeId={active}
            ariaLabel="Status"
            idBase="status"
            tabs={LONG_TABS}
            onSelect={onSelect}
          />
        </ChakraProvider>
      )
    );
  render(activeId);
  const strip = host.querySelector<HTMLElement>('[role="tablist"]')!;
  const tab = (label: string) =>
    [...strip.querySelectorAll<HTMLElement>('[role="tab"]')].find((el) => el.textContent === label)!;
  /** Wholly inside the strip, and clear of the fade at an end that still hides tabs. */
  const isInView = (element: HTMLElement) => {
    const bounds = strip.getBoundingClientRect();
    const rect = element.getBoundingClientRect();
    const hidesStart = strip.scrollLeft > 1;
    const hidesEnd = strip.scrollLeft < strip.scrollWidth - strip.clientWidth - 1;
    return (
      rect.left >= bounds.left + (hidesStart ? FADE_PX : 0) - 1 &&
      rect.right <= bounds.right - (hidesEnd ? FADE_PX : 0) + 1
    );
  };
  return { isInView, render, strip, tab };
};

it('scrolls a scrollable strip sideways instead of squeezing or wrapping tabs that do not fit', () => {
  const { strip } = renderScrollingTabs(200);
  const tabs = [...strip.querySelectorAll<HTMLElement>('[role="tab"]')];

  expect(strip.scrollWidth).toBeGreaterThan(strip.clientWidth);
  expect(host!.scrollWidth).toBe(host!.clientWidth);
  expect(new Set(tabs.map((tab) => tab.offsetTop)).size).toBe(1);
  for (const tab of tabs) {
    // A squeezed tab would clip its label: its content would be wider than its box.
    expect(tab.scrollWidth).toBeLessThanOrEqual(tab.clientWidth);
  }
  // A vertical wheel moves it sideways.
  strip.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 40 }));
  expect(strip.scrollLeft).toBe(40);
});

it('fades out whichever end of a scrolling strip hides tabs', async () => {
  const { strip } = renderScrollingTabs(200);
  const fades = async () => {
    await nextFrame();
    return {
      end: strip.hasAttribute('data-hidden-end'),
      masked: getComputedStyle(strip).maskImage !== 'none',
      start: strip.hasAttribute('data-hidden-start'),
    };
  };

  expect(await fades()).toEqual({ end: true, masked: true, start: false });
  strip.scrollLeft = 60;
  expect(await fades()).toEqual({ end: true, masked: true, start: true });
  strip.scrollLeft = strip.scrollWidth;
  expect(await fades()).toEqual({ end: false, masked: true, start: true });
});

it('brings the focused tab into view as the arrow keys move along a scrolling strip', async () => {
  const { isInView, tab } = renderScrollingTabs(200);
  act(() => tab('Alle 12.345').focus());

  for (const label of ['Aktiv 12.345', 'Erledigt 12.345', 'Fehlgeschlagen 12.345', 'Abgebrochen 12.345']) {
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(document.activeElement).toBe(tab(label));
    expect(isInView(tab(label))).toBe(true);
  }
  await act(() => userEvent.keyboard('{Home}'));
  expect(isInView(tab('Alle 12.345'))).toBe(true);
});

it('draws the focused tab ring inside the strip, which clips at the tabs height', async () => {
  const { strip, tab } = renderScrollingTabs(200);
  act(() => tab('Alle 12.345').focus());
  await act(() => userEvent.keyboard('{End}'));
  const focused = tab('Abgebrochen 12.345');
  const style = getComputedStyle(focused);
  // The ring reaches this far past the tab's border box; negative is inside it.
  const reach = Number.parseFloat(style.outlineWidth) + Number.parseFloat(style.outlineOffset);
  const bounds = strip.getBoundingClientRect();
  const rect = focused.getBoundingClientRect();

  expect(focused.matches(':focus-visible')).toBe(true);
  expect(style.outlineStyle).toBe('solid');
  // Sub-pixel tolerance: a fully scrolled strip can leave a fractional remainder.
  expect(rect.top - reach).toBeGreaterThanOrEqual(bounds.top - 0.5);
  expect(rect.bottom + reach).toBeLessThanOrEqual(bounds.bottom + 0.5);
  expect(rect.left - reach).toBeGreaterThanOrEqual(bounds.left - 0.5);
  expect(rect.right + reach).toBeLessThanOrEqual(bounds.right + 0.5);
});

it('brings a clicked, partly hidden tab wholly into view', () => {
  const onSelect = vi.fn();
  const { isInView, tab } = renderScrollingTabs(200, 'all', onSelect);
  const partlyHidden = ['Aktiv 12.345', 'Erledigt 12.345'].map(tab).find((element) => !isInView(element))!;

  // Pointer focus scrolling differs between browsers; the strip must not depend on it.
  act(() => partlyHidden.dispatchEvent(new MouseEvent('click', { bubbles: true })));

  expect(onSelect).toHaveBeenCalledOnce();
  expect(isInView(partlyHidden)).toBe(true);
});

it('brings a tab selected from outside the strip into view', () => {
  const { isInView, render, strip, tab } = renderScrollingTabs(200, 'canceled');

  // Selected on mount, at the far end.
  expect(strip.scrollLeft).toBeGreaterThan(0);
  expect(isInView(tab('Abgebrochen 12.345'))).toBe(true);
  render('all');
  expect(isInView(tab('Alle 12.345'))).toBe(true);
});

it('leaves a scrollable strip still when its tabs fit', () => {
  const { strip } = renderScrollingTabs(1000);

  expect(strip.scrollWidth).toBe(strip.clientWidth);
  expect(strip.hasAttribute('data-hidden-end')).toBe(false);
  strip.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 40 }));
  expect(strip.scrollLeft).toBe(0);
});
