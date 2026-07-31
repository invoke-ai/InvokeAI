/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryImageItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryItemActions, GalleryItemContextMenuTarget } from '@features/gallery/react';
import type { ReactNode, Ref } from 'react';

import { useGalleryItemActions } from '@features/gallery/ui/GalleryUiContext';
import { act, createRef, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GalleryImageContextMenu, GalleryItemActionsAdapter } from './GalleryImageActionsBridge';

const mocks = vi.hoisted(() => ({
  moveItemsToBoard: vi.fn(),
  useImageActions: vi.fn(),
}));

vi.mock('@workbench/image-actions', () => ({
  ImageContextMenu: ({
    target,
  }: {
    target: { itemRefs: GalleryItemRef[]; items: Array<{ kind: string; name: string }> } | null;
  }) => (
    <output data-testid="image-context-target">
      {JSON.stringify(
        target
          ? {
              itemRefs: target.itemRefs,
              items: target.items.map(({ kind, name }) => ({ kind, name })),
            }
          : null
      )}
    </output>
  ),
  useImageActions: (options: unknown) => {
    mocks.useImageActions(options);
    return {
      deleteItems: vi.fn(),
      deleteImages: vi.fn(),
      downloadItem: vi.fn(),
      downloadItems: vi.fn(),
      moveImagesToBoard: vi.fn(),
      moveItemsToBoard: mocks.moveItemsToBoard,
      openItemInNewTab: vi.fn(),
      openItemInPreview: vi.fn(),
      setItemsStarred: vi.fn(),
      setImagesStarred: vi.fn(),
    };
  },
}));

const loadedImage: GalleryImageItem = {
  boardId: 'board-a',
  category: 'general',
  createdAt: '2026-07-30T00:00:00.000Z',
  fullUrl: '/full/loaded.png',
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name: 'loaded.png',
  starred: false,
  thumbnailUrl: '/thumbnail/loaded.png',
  width: 64,
};
const mixedRefs: GalleryItemRef[] = [
  { kind: 'image', name: loadedImage.name },
  { kind: 'video', name: 'unloaded.mp4' },
];
const noop = vi.fn();

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const actionsRef = createRef<GalleryItemActions>();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ActionsProbe = ({ ref }: { ref: Ref<GalleryItemActions> }) => {
  const actions = useGalleryItemActions();
  useImperativeHandle(ref, () => actions, [actions]);
  return null;
};

const render = async (children: ReactNode) => {
  await act(() => {
    root?.render(children);
  });
};

beforeEach(() => {
  vi.clearAllMocks();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => {
    root?.unmount();
  });
  host?.remove();
  host = null;
  root = null;
});

describe('GalleryImageActionsBridge mixed selection boundaries', () => {
  it('forwards the complete canonical context target without narrowing a mixed selection', async () => {
    const target: GalleryItemContextMenuTarget = {
      itemRefs: mixedRefs,
      items: [loadedImage],
      x: 20,
      y: 40,
    };

    await render(
      <GalleryItemActionsAdapter boards={[]} generateValues={{}} projectId="project-1">
        <GalleryImageContextMenu boards={[]} target={target} onClose={noop} />
      </GalleryItemActionsAdapter>
    );

    expect(host?.querySelector('[data-testid="image-context-target"]')?.textContent).toBe(
      JSON.stringify({
        itemRefs: mixedRefs,
        items: [{ kind: 'image', name: loadedImage.name }],
      })
    );
  });

  it('forwards an ordered mixed board move through the common action port', async () => {
    await render(
      <GalleryItemActionsAdapter boards={[]} generateValues={{}} projectId="project-1">
        <ActionsProbe ref={actionsRef} />
      </GalleryItemActionsAdapter>
    );

    await act(async () => {
      await actionsRef.current?.moveItemsToBoard(mixedRefs, 'board-b');
    });

    expect(mocks.moveItemsToBoard).toHaveBeenCalledWith(mixedRefs, 'board-b');
  });

  it('forwards the live gallery action context getter used by guarded delete successor selection', async () => {
    const getItemActionContext = vi.fn(() => null);

    await render(
      <GalleryItemActionsAdapter
        boards={[]}
        generateValues={{}}
        getItemActionContext={getItemActionContext}
        projectId="project-1"
      >
        <ActionsProbe ref={actionsRef} />
      </GalleryItemActionsAdapter>
    );

    expect(mocks.useImageActions).toHaveBeenCalledWith(
      expect.objectContaining({
        getItemActionContext,
      })
    );
  });
});
