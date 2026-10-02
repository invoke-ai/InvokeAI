import type { FloatingWidgetState } from '@workbench/layoutContracts';

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * Raising a window changes how the layer stacks it. It must not change where the window sits in the DOM: moving
 * a node drops keyboard focus from it, and focus arriving in a window is exactly what raises it.
 */

const layerMocks = vi.hoisted(() => ({
  project: { floatingWidgets: {} as Record<string, FloatingWidgetState>, id: 'project-1' },
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  shallowEqual: Object.is,
  useActiveProjectSelector: (selector: (project: typeof layerMocks.project) => unknown) => selector(layerMocks.project),
}));

vi.mock('./FloatingWidgetWindow', () => ({
  FloatingWidgetWindow: ({ instanceId, stackRank }: { instanceId: string; stackRank: number }) => (
    <button data-rank={stackRank} data-window={instanceId} type="button">
      {instanceId}
    </button>
  ),
}));

import { FloatingWidgetLayer } from './FloatingWidgetLayer';

const windowState = (stackOrder: number): FloatingWidgetState => ({
  heightPx: 300,
  mode: 'windowed',
  returnIndex: 0,
  returnRegion: 'right',
  stackOrder,
  widthPx: 300,
  x: 0,
  y: 0,
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (floatingWidgets: Record<string, FloatingWidgetState>) => {
  layerMocks.project = { ...layerMocks.project, floatingWidgets };
  await act(() => {
    root?.render(<FloatingWidgetLayer />);
  });
  // The window chunk is lazy: let its import settle and the suspended layer commit.
  await vi.waitFor(async () => {
    await act(async () => {
      await Promise.resolve();
    });
    expect(host!.querySelector('[data-window]')).not.toBeNull();
  });
};
const windows = () =>
  [...host!.querySelectorAll<HTMLElement>('[data-window]')].map((element) => [
    element.dataset.window,
    element.dataset.rank,
  ]);

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('FloatingWidgetLayer', () => {
  it('restacks a raised window without moving it in the DOM, so it keeps keyboard focus', async () => {
    await render({ gallery: windowState(2), queue: windowState(1) });

    expect(windows()).toEqual([
      ['gallery', '1'],
      ['queue', '0'],
    ]);

    const queue = host!.querySelector<HTMLElement>('[data-window="queue"]')!;
    queue.focus();
    // Focus arriving in the lower window raises it.
    await render({ gallery: windowState(1), queue: windowState(2) });

    expect(windows()).toEqual([
      ['gallery', '0'],
      ['queue', '1'],
    ]);
    expect(host!.querySelector('[data-window="queue"]')).toBe(queue);
    expect(document.activeElement).toBe(queue);
  });
});
