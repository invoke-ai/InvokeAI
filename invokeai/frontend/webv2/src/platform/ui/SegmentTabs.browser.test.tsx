import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { SegmentTabs } from './SegmentTabs';

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
