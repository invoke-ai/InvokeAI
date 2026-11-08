/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * Share rail visibility/progress coverage across topbar and floating Preview: offline/idle hides, unstepped runs
 * sweep, stepped runs fill.
 */

const state = vi.hoisted(() => ({
  activeItemIds: [] as number[],
  isConnected: true,
  percentage: null as number | null,
  queueItems: [] as { backendItemIds: number[]; id: string; status: string }[],
}));

vi.mock('@features/models', () => ({ useModelLoads: () => [] }));
vi.mock('@features/queue/react', () => ({
  useActiveProgressItemIds: () => state.activeItemIds,
  useItemProgress: (itemId: number | null) => (itemId === null ? null : { percentage: state.percentage }),
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (select: (project: unknown) => unknown) => select({ queue: { items: state.queueItems } }),
  useWorkbenchSelector: (select: (snapshot: unknown) => unknown) =>
    select({ backendConnection: { status: state.isConnected ? 'connected' : 'disconnected' } }),
}));

import { QueueProgressRail } from './QueueProgressRail';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async () => {
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <QueueProgressRail css={{ height: '3px', position: 'absolute', top: 0, insetInline: 0 }} />
      </ChakraProvider>
    );
    await Promise.resolve();
  });

  return host?.querySelector<HTMLElement>('[data-queue-progress-rail]') ?? null;
};

const runningItem = { backendItemIds: [7], id: 'item-7', status: 'running' };

beforeEach(() => {
  state.activeItemIds = [];
  state.isConnected = true;
  state.percentage = null;
  state.queueItems = [];
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

describe('QueueProgressRail', () => {
  it('renders nothing while the queue is idle', async () => {
    expect(await render()).toBeNull();
  });

  it('renders nothing while the backend is away, even with open work', async () => {
    state.queueItems = [runningItem];
    state.activeItemIds = [7];
    state.isConnected = false;

    expect(await render()).toBeNull();
  });

  it('sweeps while the running item has not reported a step, then fills to its fraction', async () => {
    state.queueItems = [runningItem];
    state.activeItemIds = [7];

    const sweeping = await render();
    expect(sweeping).not.toBeNull();
    expect(sweeping?.querySelector('[style*="width"]')).toBeNull();

    state.percentage = 0.4;
    const filling = await render();
    const fill = filling?.querySelector<HTMLElement>('[style*="width"]');
    expect(fill?.style.width).toBe('40%');
  });
});
