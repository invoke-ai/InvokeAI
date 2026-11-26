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

import { CanvasScrollInputFeed } from './CanvasScrollInputFeed';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const STORAGE_KEY = 'invokeai:v7:webv2:canvas-trackpad-detected';

let root: Root | null = null;
let host: HTMLDivElement | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
  harness.store = null;
  localStorage.removeItem(STORAGE_KEY);
});

const mount = async () => {
  const store = createWorkbenchStore();
  harness.store = store;
  const stores = createEngineStores();
  const engine = {
    interaction: {
      get: (key: 'scrollInput' | 'trackpadDetected') => stores[key].get(),
      set: (key: 'scrollInput' | 'trackpadDetected', value: never) => stores[key].set(value),
      subscribe: (key: 'scrollInput' | 'trackpadDetected', listener: () => void) => stores[key].subscribe(listener),
    },
  } as never;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => root!.render(<CanvasScrollInputFeed engine={engine} projectId={store.getState().activeProjectId} />));
  return { store, stores };
};

describe('CanvasScrollInputFeed', () => {
  it('feeds the project’s scroll input to the engine', async () => {
    const { store, stores } = await mount();
    expect(stores.scrollInput.get()).toBe('auto');
    await act(() => store.commands.widgets.patchValues('canvas', { scrollInput: 'trackpad' }));
    expect(stores.scrollInput.get()).toBe('trackpad');
  });

  it('persists a detected trackpad for this browser and restores it on the next mount', async () => {
    const first = await mount();
    expect(first.stores.trackpadDetected.get()).toBe(false);
    first.stores.trackpadDetected.set(true);
    expect(localStorage.getItem(STORAGE_KEY)).toBe('true');
    await act(() => root?.unmount());
    host?.remove();

    const second = await mount();
    expect(second.stores.trackpadDetected.get()).toBe(true);
  });
});
