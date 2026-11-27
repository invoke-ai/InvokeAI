/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryImageItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryItemContextMenuTarget } from '@features/gallery/react';
import type { ReactNode } from 'react';

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GalleryImageContextMenu, GalleryItemActionsAdapter } from './GalleryImageActionsBridge';

const mocks = vi.hoisted(() => ({
  requestDeletionConfirmation: vi.fn(),
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
  useDeletionConfirmation: () => ({
    dialog: <output data-testid="deletion-confirmation-dialog" />,
    requestDeletionConfirmation: mocks.requestDeletionConfirmation,
  }),
  useImageActions: (options: unknown) => {
    mocks.useImageActions(options);
    return {
      deleteItems: vi.fn(),
      deleteImages: vi.fn(),
      downloadItem: vi.fn(),
      downloadItems: vi.fn(),
      moveImagesToBoard: vi.fn(),
      moveItemsToBoard: vi.fn(),
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
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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
  it('wires the shared deletion-confirmation gate into the actions adapter', async () => {
    await render(
      <GalleryItemActionsAdapter boards={[]} generateValues={{}} projectId="project-1">
        <span />
      </GalleryItemActionsAdapter>
    );

    expect(host?.querySelector('[data-testid="deletion-confirmation-dialog"]')).not.toBeNull();
    expect(mocks.useImageActions).toHaveBeenCalledWith(
      expect.objectContaining({ requestDeletionConfirmation: mocks.requestDeletionConfirmation })
    );
  });

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
});
