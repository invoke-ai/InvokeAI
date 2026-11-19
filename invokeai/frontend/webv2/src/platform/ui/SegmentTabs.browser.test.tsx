import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { SegmentTabs, type SegmentTab } from './SegmentTabs';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const noop = (): void => {};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
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

it('brings a clicked, partly hidden tab wholly into view', () => {
  const onSelect = vi.fn();
  const { isInView, tab } = renderScrollingTabs(200, 'all', onSelect);
  const partlyHidden = ['Aktiv 12.345', 'Erledigt 12.345'].map(tab).find((element) => !isInView(element))!;

  // Pointer focus scrolling differs between browsers; the strip must not depend on it.
  act(() => partlyHidden.dispatchEvent(new MouseEvent('click', { bubbles: true })));

  expect(onSelect).toHaveBeenCalledOnce();
  expect(isInView(partlyHidden)).toBe(true);
});
