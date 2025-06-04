import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import type { S } from 'services/api/types';

export const useStagingAreaKeyboardNav = (
  items: S['SessionQueueItem'][],
  selectedItemId: number | null,
  onSelectItemId: (item_id: number) => void
) => {
  const onNext = useCallback(() => {
    if (selectedItemId === null) {
      return;
    }
    const currentIndex = items.findIndex((item) => item.item_id === selectedItemId);
    const nextIndex = (currentIndex + 1) % items.length;
    const nextItem = items[nextIndex];
    if (!nextItem) {
      return;
    }
    onSelectItemId(nextItem.item_id);
  }, [items, onSelectItemId, selectedItemId]);
  const onPrev = useCallback(() => {
    if (selectedItemId === null) {
      return;
    }
    const currentIndex = items.findIndex((item) => item.item_id === selectedItemId);
    const prevIndex = (currentIndex - 1 + items.length) % items.length;
    const prevItem = items[prevIndex];
    if (!prevItem) {
      return;
    }
    onSelectItemId(prevItem.item_id);
  }, [items, onSelectItemId, selectedItemId]);

  const onFirst = useCallback(() => {
    const first = items.at(0);
    if (!first) {
      return;
    }
    onSelectItemId(first.item_id);
  }, [items, onSelectItemId]);
  const onLast = useCallback(() => {
    const last = items.at(-1);
    if (!last) {
      return;
    }
    onSelectItemId(last.item_id);
  }, [items, onSelectItemId]);

  useHotkeys('left', onPrev, { preventDefault: true });
  useHotkeys('right', onNext, { preventDefault: true });
  useHotkeys('meta+left', onFirst, { preventDefault: true });
  useHotkeys('meta+right', onLast, { preventDefault: true });
};
