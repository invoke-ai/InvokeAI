import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ImageActions } from './useImageActions';

import { useImageActions } from './useImageActions';

const mocks = vi.hoisted(() => ({
  addToBoard: vi.fn(),
  deleteImages: vi.fn(),
  galleryPatchItems: vi.fn(),
  invalidateGallery: vi.fn(),
  invalidateGalleryItems: vi.fn(),
  openWorkbenchWidget: vi.fn(),
  patchGalleryItemCaches: vi.fn((..._args: unknown[]) => vi.fn()),
  removeFromBoard: vi.fn(),
  reportError: vi.fn(),
  setStarred: vi.fn(),
}));

vi.mock('@features/gallery', () => ({
  galleryImages: { metadata: vi.fn() },
  galleryOrganization: {
    addToBoard: (...args: unknown[]) => mocks.addToBoard(...args),
    deleteImages: (...args: unknown[]) => mocks.deleteImages(...args),
    removeFromBoard: (...args: unknown[]) => mocks.removeFromBoard(...args),
    setStarred: (...args: unknown[]) => mocks.setStarred(...args),
  },
  galleryTransfers: { downloadArchive: vi.fn() },
  toGalleryItemKey: ({ kind, name }: { kind: string; name: string }) => `${kind}:${name}`,
}));

vi.mock('@features/gallery/queries', () => ({
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
  invalidateGalleryItems: (...args: unknown[]) => mocks.invalidateGalleryItems(...args),
  patchGalleryItemCaches: (...args: unknown[]) => mocks.patchGalleryItemCaches(...args),
}));

vi.mock('@features/models', () => ({
  ensureModelsLoaded: vi.fn(() => Promise.resolve()),
  getModelsSnapshot: () => ({ models: [] }),
  useModelsSelector: (selector: (snapshot: { models: never[] }) => unknown) => selector({ models: [] }),
}));

vi.mock('@workbench/useOpenWorkbenchWidget', () => ({
  useOpenWorkbenchWidget: () => mocks.openWorkbenchWidget,
}));

vi.mock('@workbench/canvas-operations/api', () => ({
  getCanvasEngine: vi.fn(),
  getCanvasImportNotice: vi.fn(),
  importGalleryImagesToCanvas: vi.fn(),
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    canvas: { apply: vi.fn() },
    gallery: {
      patchItems: (...args: unknown[]) => mocks.galleryPatchItems(...args),
      removeItems: vi.fn(),
      selectImage: vi.fn(),
      setCompareImage: vi.fn(),
    },
    generation: { patchSettings: vi.fn() },
    notifications: { add: vi.fn(), reportError: (...args: unknown[]) => mocks.reportError(...args) },
  }),
  useWorkbenchQueries: () => ({
    getProject: vi.fn(),
    getSnapshot: () => ({ activeProject: { id: 'project-1' }, projects: [] }),
    isActiveProject: vi.fn(() => true),
  }),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const actionsRef = createRef<ImageActions>();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Probe = ({ ref }: { ref: Ref<ImageActions> }) => {
  const actions = useImageActions({
    boards: [],
    generateValues: {},
    projectId: 'project-1',
  });

  useImperativeHandle(ref, () => actions, [actions]);

  return null;
};

beforeEach(async () => {
  vi.clearAllMocks();
  accountLifecycle.activate('user-a');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <Probe ref={actionsRef} />
      </QueryClientProvider>
    );
  });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('partial image mutation outcomes', () => {
  it('moves and patches only images confirmed by the backend', async () => {
    mocks.addToBoard.mockResolvedValue(['moved.png']);

    await act(async () => {
      await actionsRef.current?.moveImagesToBoard(['moved.png', 'locked.png'], 'board-1');
    });

    const result = {
      failed: [{ kind: 'image', name: 'locked.png' }],
      succeeded: [{ kind: 'image', name: 'moved.png' }],
    };

    expect(mocks.patchGalleryItemCaches).toHaveBeenCalledWith(expect.anything(), {
      boardId: 'board-1',
      kind: 'move',
      result,
    });
    expect(mocks.galleryPatchItems).toHaveBeenCalledWith(['image:moved.png'], { boardId: 'board-1' });
  });

  it('stars and patches only images confirmed by the backend', async () => {
    mocks.setStarred.mockResolvedValue(['starred.png']);

    await act(async () => {
      await actionsRef.current?.setImagesStarred(['starred.png', 'locked.png'], true);
    });

    const result = {
      failed: [{ kind: 'image', name: 'locked.png' }],
      succeeded: [{ kind: 'image', name: 'starred.png' }],
    };

    expect(mocks.patchGalleryItemCaches).toHaveBeenCalledWith(expect.anything(), {
      kind: 'star',
      result,
      starred: true,
    });
    expect(mocks.galleryPatchItems).toHaveBeenCalledWith(['image:starred.png'], { starred: true });
  });
});
