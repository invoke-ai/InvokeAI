import type { GalleryUiAdapter } from '@features/gallery/react';
import type { Project } from '@workbench/projectContracts';
import type { ReactNode } from 'react';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GalleryUiAdapterProvider } from './GalleryUiAdapter';

let store: ReturnType<typeof createWorkbenchStore>;
let adapter: GalleryUiAdapter;
let renderCount: number;
let host: HTMLDivElement;
let root: Root;
const noop = () => undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('@features/gallery/react', () => ({
  GalleryUiProvider: ({ adapter: next, children }: { adapter: GalleryUiAdapter; children: ReactNode }) => {
    adapter = next;
    renderCount += 1;
    return children;
  },
}));
vi.mock('@workbench/widgets/preview/livePreviewFollow', () => ({
  useLivePreviewFollow: () => ({
    sessions: [],
    gallerySessions: [],
    pinnedSessionId: null,
    followedSessionId: null,
    follow: vi.fn(),
    pin: vi.fn(),
    showAll: vi.fn(),
  }),
}));
vi.mock('@workbench/projects/useProjectFileActions', () => ({ useExportLibraryProject: () => noop }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => noop }));
vi.mock('./GalleryImageActionsBridge', () => ({
  GalleryItemActionsAdapter: () => null,
  GalleryImageContextMenu: () => null,
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) =>
    useExternalStoreSelector(
      store.subscribe,
      store.getSnapshot,
      (snapshot) => selector(snapshot.activeProject),
      shallowEqual
    ),
  useWorkbenchCommands: () => store.commands,
  useWorkbenchQueries: () => store.queries,
}));

const renderAdapter = async (absent: string[] = []) => {
  const state = createInitialWorkbenchState();
  const project = state.projects[0]!;
  project.widgetInstances = Object.fromEntries(
    Object.entries(project.widgetInstances).filter(([, instance]) => !absent.includes(instance.typeId))
  );
  store = createWorkbenchStore(state);
  await act(() => root.render(<GalleryUiAdapterProvider>{null}</GalleryUiAdapterProvider>));
};

beforeEach(() => {
  accountLifecycle.activate('gallery-subscription-test');
  renderCount = 0;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

describe('Gallery adapter subscriptions', () => {
  it.each([['gallery'], ['generate'], ['gallery', 'generate']])(
    'retains the adapter during unrelated layout updates with absent widgets %j',
    async (...absent) => {
      await renderAdapter(absent);
      const previousAdapter = adapter;
      const previousRenderCount = renderCount;
      const previousSnapshot = store.getSnapshot();
      const nextWidth = previousSnapshot.activeProject.widgetRegions.left.sizePx + 20;

      await act(() => store.commands.layout.setRegionSize('left', nextWidth));

      expect(store.getSnapshot()).not.toBe(previousSnapshot);
      expect(store.getSnapshot().activeProject.widgetRegions.left.sizePx).toBe(nextWidth);
      expect(adapter).toBe(previousAdapter);
      expect(renderCount).toBe(previousRenderCount);
    }
  );

  it('publishes relevant Gallery and Generate value changes through the subscribed adapter', async () => {
    await renderAdapter();
    const previousAdapter = adapter;

    await act(() => store.commands.gallery.updateSettings({ imageDensityPercent: 24 }));

    expect(adapter).not.toBe(previousAdapter);
    expect(adapter.galleryValues.imageDensityPercent).toBe(24);
    const galleryAdapter = adapter;

    await act(() => store.commands.widgets.patchValues('generate', { positivePrompt: 'Updated prompt' }));

    expect(adapter).not.toBe(galleryAdapter);
    expect(adapter.generateValues.positivePrompt).toBe('Updated prompt');
    expect(adapter.galleryValues).toBe(galleryAdapter.galleryValues);
  });
});
