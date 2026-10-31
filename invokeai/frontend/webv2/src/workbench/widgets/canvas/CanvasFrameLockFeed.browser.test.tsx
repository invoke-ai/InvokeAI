/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({ store: null as ReturnType<typeof createWorkbenchStore> | null }));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useWorkbenchQueries: () => harness.store!.queries,
  useWorkbenchSubscription: () => harness.store!.subscribe,
}));

import { CanvasFrameLockFeed } from './CanvasFrameLockFeed';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLDivElement | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
  harness.store = null;
});

const mount = async () => {
  const store = createWorkbenchStore();
  harness.store = store;
  const stores = createEngineStores();
  const engine = {
    interaction: {
      get: (key: 'bboxOptions') => stores[key].get(),
      set: (key: 'bboxOptions', value: { aspectLocked: boolean; aspectRatio: number }) => stores[key].set(value),
    },
  } as never;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => root!.render(<CanvasFrameLockFeed engine={engine} projectId={store.getState().activeProjectId} />));
  return { bboxOptions: () => stores.bboxOptions.get(), store };
};

describe('CanvasFrameLockFeed', () => {
  it('gives the Frame tool Generate’s own lock while the canvas drives Generate, and leaves it alone otherwise', async () => {
    const { bboxOptions, store } = await mount();
    store.commands.generation.setSource('generate');
    store.commands.generation.patchSettings({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 16 / 9,
    });
    expect(bboxOptions()).toEqual({ aspectLocked: false, aspectRatio: 1 });

    await act(() => store.commands.generation.setSource('canvas'));
    expect(bboxOptions()).toEqual({ aspectLocked: true, aspectRatio: 16 / 9 });

    // A preset label alone (no lock of the person's) leaves the frame free.
    await act(() =>
      store.commands.generation.patchSettings({ aspectRatioId: '1:1', aspectRatioIsLocked: false, aspectRatioValue: 1 })
    );
    expect(bboxOptions()).toEqual({ aspectLocked: false, aspectRatio: 1 });
  });
});
