import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

export const useStagingAreaKeyboardNav = () => {
  const ctx = useCanvasSessionContext();
  const onNext = useCallback(() => {
    const selectedItemId = ctx.$selectedItemId.get();
    if (selectedItemId === null) {
      return;
    }
    const items = ctx.$items.get();
    const currentIndex = items.findIndex((item) => item.item_id === selectedItemId);
    const nextIndex = (currentIndex + 1) % items.length;
    const nextItem = items[nextIndex];
    if (!nextItem) {
      return;
    }
    ctx.$selectedItemId.set(nextItem.item_id);
  }, [ctx.$items, ctx.$selectedItemId]);
  const onPrev = useCallback(() => {
    const selectedItemId = ctx.$selectedItemId.get();
    if (selectedItemId === null) {
      return;
    }
    const items = ctx.$items.get();
    const currentIndex = items.findIndex((item) => item.item_id === selectedItemId);
    const prevIndex = (currentIndex - 1 + items.length) % items.length;
    const prevItem = items[prevIndex];
    if (!prevItem) {
      return;
    }
    ctx.$selectedItemId.set(prevItem.item_id);
  }, [ctx.$items, ctx.$selectedItemId]);

  const onFirst = useCallback(() => {
    const items = ctx.$items.get();
    const first = items.at(0);
    if (!first) {
      return;
    }
    ctx.$selectedItemId.set(first.item_id);
  }, [ctx.$items, ctx.$selectedItemId]);
  const onLast = useCallback(() => {
    const items = ctx.$items.get();
    const last = items.at(-1);
    if (!last) {
      return;
    }
    ctx.$selectedItemId.set(last.item_id);
  }, [ctx.$items, ctx.$selectedItemId]);

  useHotkeys('left', onPrev, { preventDefault: true });
  useHotkeys('right', onNext, { preventDefault: true });
  useHotkeys('meta+left', onFirst, { preventDefault: true });
  useHotkeys('meta+right', onLast, { preventDefault: true });
};
