/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryImageItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryImageActions, GalleryItemContextMenuTarget } from '@features/gallery/react';
import type { ReactNode, Ref } from 'react';

import { useGalleryImageActions } from '@features/gallery/ui/GalleryUiContext';
import { act, createRef, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GalleryImageActionsAdapter, GalleryImageContextMenu } from './GalleryImageActionsBridge';

const mocks = vi.hoisted(() => ({
  moveImagesToBoard: vi.fn(),
}));

vi.mock('@workbench/image-actions', () => ({
  ImageContextMenu: ({ target }: { target: { images: Array<{ imageName: string }> } | null }) => (
    <output data-testid="image-context-target">
      {JSON.stringify(target?.images.map((image) => image.imageName) ?? null)}
    </output>
  ),
  useImageActions: () => ({
    deleteImages: vi.fn(),
    moveImagesToBoard: mocks.moveImagesToBoard,
    setImagesStarred: vi.fn(),
  }),
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
const directActions: GalleryImageActions = {
  deleteImages: vi.fn(),
  moveImagesToBoard: mocks.moveImagesToBoard,
  moveItemsToBoard: vi.fn(),
  setImagesStarred: vi.fn(),
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const actionsRef = createRef<GalleryImageActions>();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ActionsProbe = ({ ref }: { ref: Ref<GalleryImageActions> }) => {
  const actions = useGalleryImageActions();
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
  it('does not expose a partial image context target when an unloaded video ref is selected', async () => {
    const target: GalleryItemContextMenuTarget = {
      itemRefs: mixedRefs,
      items: [loadedImage],
      x: 20,
      y: 40,
    };

    await render(<GalleryImageContextMenu actions={directActions} boards={[]} target={target} onClose={noop} />);

    expect(host?.querySelector('[data-testid="image-context-target"]')?.textContent).toBe('null');
  });

  it('does not move the loaded image subset of an image plus unloaded-video ref vector', async () => {
    await render(
      <GalleryImageActionsAdapter boards={[]} generateValues={{}} projectId="project-1" onImagesDeleted={noop}>
        <ActionsProbe ref={actionsRef} />
      </GalleryImageActionsAdapter>
    );

    await act(async () => {
      await actionsRef.current?.moveItemsToBoard(mixedRefs, 'board-b');
    });

    expect(mocks.moveImagesToBoard).not.toHaveBeenCalled();
  });
});
