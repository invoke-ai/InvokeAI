/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryBoardDragMonitor } from './GalleryBoardDragMonitor';
import { getGalleryBoardDropData, getGalleryItemDragData } from './galleryDnd';
import { GalleryWidgetContext } from './GalleryWidgetContext';

const dnd = vi.hoisted(() => ({ monitor: null as null | Record<string, (event: never) => void> }));

vi.mock('@dnd-kit/core', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useDndMonitor: (monitor: Record<string, (event: never) => void>) => {
    dnd.monitor = monitor;
  },
}));

const updateSettings = vi.fn();
const moveItemsToBoard = vi.fn(async () => {});
const draggedRef = { kind: 'image' as const, name: 'dragged.png' };
const draggedItem = {
  boardId: 'source-board',
  category: 'general' as const,
  createdAt: '2026-07-30T12:00:00.000Z',
  fullUrl: '/images/dragged.png/full',
  height: 64,
  isIntermediate: false,
  kind: 'image' as const,
  name: 'dragged.png',
  starred: false,
  thumbnailUrl: '/images/dragged.png/thumbnail',
  width: 64,
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderMonitor = async (boardPanelCollapsed: boolean) => {
  const contextValue = {
    actions: { updateSettings },
    gallery: {
      items: [draggedItem],
      settings: { boardPanelCollapsed },
    },
    itemActions: { moveItemsToBoard },
  } as unknown as GalleryWidgetContextValue;

  await act(() => {
    root?.render(
      <GalleryWidgetContext value={contextValue}>
        <GalleryBoardDragMonitor />
      </GalleryWidgetContext>
    );
  });
};

const dragStart = (data: unknown) =>
  act(() => dnd.monitor?.onDragStart?.({ active: { data: { current: data } } } as never));

const dragEnd = (activeData: unknown, overData?: unknown) =>
  act(() =>
    dnd.monitor?.onDragEnd?.({
      active: { data: { current: activeData } },
      over: overData ? { data: { current: overData } } : null,
    } as never)
  );

const dragCancel = () => act(() => dnd.monitor?.onDragCancel?.({} as never));

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  dnd.monitor = null;
  vi.clearAllMocks();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GalleryBoardDragMonitor', () => {
  it('temporarily expands a collapsed panel, forwards the drop, and restores collapse', async () => {
    await renderMonitor(true);
    const dragData = getGalleryItemDragData([draggedRef]);

    dragStart(dragData);
    expect(updateSettings).toHaveBeenNthCalledWith(1, { boardPanelCollapsed: false });

    dragEnd(dragData, getGalleryBoardDropData('target-board', 'board'));

    expect(moveItemsToBoard).toHaveBeenCalledExactlyOnceWith([draggedRef], 'target-board');
    expect(updateSettings).toHaveBeenNthCalledWith(2, { boardPanelCollapsed: true });
  });

  it('leaves a panel that was already expanded open', async () => {
    await renderMonitor(false);
    const dragData = getGalleryItemDragData([draggedRef]);

    dragStart(dragData);
    dragEnd(dragData, getGalleryBoardDropData('target-board', 'board'));

    expect(moveItemsToBoard).toHaveBeenCalledOnce();
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it('ignores non-gallery drags', async () => {
    await renderMonitor(true);

    dragStart({ kind: 'workflow-node' });
    dragEnd({ kind: 'workflow-node' }, getGalleryBoardDropData('target-board', 'board'));

    expect(updateSettings).not.toHaveBeenCalled();
    expect(moveItemsToBoard).not.toHaveBeenCalled();
  });

  it('restores collapse after cancellation', async () => {
    await renderMonitor(true);

    dragStart(getGalleryItemDragData([draggedRef]));
    dragCancel();

    expect(updateSettings).toHaveBeenLastCalledWith({ boardPanelCollapsed: true });
    expect(moveItemsToBoard).not.toHaveBeenCalled();
  });
});
