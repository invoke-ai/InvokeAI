import { act, useEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  resolveMany: vi.fn(),
  selectBoard: vi.fn(),
  selectItem: vi.fn(),
}));

vi.mock('@features/gallery', () => ({
  galleryImages: { resolveMany: mocks.resolveMany },
  legacyGeneratedImageToGalleryItem: (image: { image_name: string }) => image,
  toGalleryItemKey: (key: { kind: string; name: string }) => key,
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    gallery: { selectBoard: mocks.selectBoard, selectItem: mocks.selectItem },
  }),
}));

import { useMapSelection } from './useSelectMapImage';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

/** Published from an effect, not during render, so the probe stays render-pure. */
const handle: { click: ((imageName: string) => void) | null } = { click: null };

const Probe = () => {
  const { selectImage } = useMapSelection();

  useEffect(() => {
    handle.click = selectImage;
  }, [selectImage]);

  return null;
};

/** Runs `fn` inside act and drains the microtask queue so promise callbacks land. */
const flush = async (fn: () => void = () => {}) => {
  await act(async () => {
    fn();
    await Promise.resolve();
  });
};

const mount = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await flush(() => root?.render(<Probe />));
};

const unmount = async () => {
  await flush(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
  handle.click = null;
};

/** A promise plus the trigger that settles it, so click ordering can be forced. */
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });

  return { promise, resolve };
};

afterEach(async () => {
  if (root) {
    await unmount();
  }
  mocks.resolveMany.mockReset();
  mocks.selectBoard.mockReset();
  mocks.selectItem.mockReset();
});

describe('useMapSelection', () => {
  it('dispatches the selection for a click', async () => {
    mocks.resolveMany.mockResolvedValue([{ image_name: 'a.png' }]);
    await mount();

    await flush(() => handle.click?.('a.png'));

    expect(mocks.selectItem).toHaveBeenCalledTimes(1);
    expect(mocks.selectItem.mock.calls[0]?.[0]).toEqual({ image_name: 'a.png' });
  });

  it("selects the image's board before the image itself", async () => {
    // The map spans every accessible board, but selectGalleryItem stamps the
    // navigation query from whatever list the gallery is currently showing. A
    // cross-board click without this left that query describing a list the
    // image was never in, and Preview's next/prev found no cursor and went
    // dead until the user re-selected from the grid.
    mocks.resolveMany.mockResolvedValue([{ boardId: 'board-portraits', image_name: 'a.png' }]);
    await mount();

    await flush(() => handle.click?.('a.png'));

    expect(mocks.selectBoard).toHaveBeenCalledWith('board-portraits');
    expect(mocks.selectBoard.mock.invocationCallOrder[0]).toBeLessThan(mocks.selectItem.mock.invocationCallOrder[0]);
  });

  it('does not touch the board for a click that never resolves an image', async () => {
    mocks.resolveMany.mockResolvedValue([]);
    await mount();

    await flush(() => handle.click?.('gone.png'));

    expect(mocks.selectBoard).not.toHaveBeenCalled();
    expect(mocks.selectItem).not.toHaveBeenCalled();
  });

  it('ignores a slow click that resolves after a newer one', async () => {
    const slow = deferred<{ image_name: string }[]>();
    const fast = deferred<{ image_name: string }[]>();

    mocks.resolveMany.mockReturnValueOnce(slow.promise).mockReturnValueOnce(fast.promise);
    await mount();

    await flush(() => {
      handle.click?.('slow.png');
      handle.click?.('fast.png');
    });

    await flush(() => {
      fast.resolve([{ image_name: 'fast.png' }]);
      slow.resolve([{ image_name: 'slow.png' }]);
    });

    // Only the most recent click may win, regardless of resolution order.
    expect(mocks.selectItem.mock.calls.map((call) => call[0].image_name)).toEqual(['fast.png']);
  });

  it('ignores a click left in flight across an unmount/remount', async () => {
    // The regression this guards: a per-mount counter is reset by the remount,
    // so the abandoned click compares against a dead counter, passes, and
    // overwrites the newer mount's selection. Reachable by switching the right
    // panel away from the map and back while a hydrate is in flight.
    const stale = deferred<{ image_name: string }[]>();
    const fresh = deferred<{ image_name: string }[]>();

    mocks.resolveMany.mockReturnValueOnce(stale.promise).mockReturnValueOnce(fresh.promise);

    await mount();
    await flush(() => handle.click?.('stale.png'));
    await unmount();

    await mount();
    await flush(() => handle.click?.('fresh.png'));

    await flush(() => {
      fresh.resolve([{ image_name: 'fresh.png' }]);
      stale.resolve([{ image_name: 'stale.png' }]);
    });

    expect(mocks.selectItem.mock.calls.map((call) => call[0].image_name)).toEqual(['fresh.png']);
  });

  it('leaves the selection alone when hydrate fails or the image is gone', async () => {
    mocks.resolveMany.mockRejectedValueOnce(new Error('deleted')).mockResolvedValueOnce([]);
    await mount();

    await flush(() => handle.click?.('gone.png'));
    await flush(() => handle.click?.('missing.png'));

    expect(mocks.selectItem).not.toHaveBeenCalled();
  });
});
