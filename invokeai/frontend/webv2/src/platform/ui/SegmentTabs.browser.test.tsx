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
