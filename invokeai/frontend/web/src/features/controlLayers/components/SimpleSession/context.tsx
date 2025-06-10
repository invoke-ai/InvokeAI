import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppStore } from 'app/store/nanostores/store';
import { getOutputImageName } from 'features/controlLayers/components/SimpleSession/shared';
import type { ProgressImage } from 'features/nodes/types/common';
import type { Atom, WritableAtom } from 'nanostores';
import { atom, computed, effect } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { queueApi } from 'services/api/endpoints/queue';
import type { S } from 'services/api/types';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';
import { z } from 'zod';

export const zAutoSwitchMode = z.enum(['off', 'first_progress', 'completed']);
export type AutoSwitchMode = z.infer<typeof zAutoSwitchMode>;

export type ProgressData = {
  itemId: number;
  progressEvent: S['InvocationProgressEvent'] | null;
  progressImage: ProgressImage | null;
  outputImageName: string | null;
};

const getInitialProgressData = (itemId: number): ProgressData => ({
  itemId,
  progressEvent: null,
  progressImage: null,
  outputImageName: null,
});

export const useProgressData = (
  $progressData: WritableAtom<Record<number, ProgressData>>,
  itemId: number
): ProgressData => {
  const [value, setValue] = useState<ProgressData>(() => {
    return $progressData.get()[itemId] ?? getInitialProgressData(itemId);
  });
  useEffect(() => {
    const unsub = $progressData.subscribe((data) => {
      const progressData = data[itemId];
      if (!progressData) {
        return;
      }
      setValue(progressData);
    });
    return () => {
      unsub();
    };
  }, [$progressData, itemId]);

  return value;
};

const setProgress = ($progressData: WritableAtom<Record<number, ProgressData>>, data: S['InvocationProgressEvent']) => {
  const progressData = $progressData.get();
  const current = progressData[data.item_id];
  if (current) {
    const next = { ...current };
    next.progressEvent = data;
    if (data.image) {
      next.progressImage = data.image;
    }
    $progressData.set({
      ...progressData,
      [data.item_id]: next,
    });
  } else {
    $progressData.set({
      ...progressData,
      [data.item_id]: {
        itemId: data.item_id,
        progressEvent: data,
        progressImage: data.image ?? null,
        outputImageName: null,
      },
    });
  }
};

type CanvasSessionContextValue = {
  session: { id: string; type: 'simple' | 'advanced' };
  $items: Atom<S['SessionQueueItem'][]>;
  $itemCount: Atom<number>;
  $hasItems: Atom<boolean>;
  $progressData: WritableAtom<Record<string, ProgressData>>;
  $selectedItemId: WritableAtom<number | null>;
  $selectedItem: Atom<S['SessionQueueItem'] | null>;
  $selectedItemIndex: Atom<number | null>;
  $selectedItemOutputImageName: Atom<string | null>;
  $autoSwitch: WritableAtom<AutoSwitchMode>;
  $lastLoadedItemId: WritableAtom<number | null>;
  selectNext: () => void;
  selectPrev: () => void;
  selectFirst: () => void;
  selectLast: () => void;
};

const CanvasSessionContext = createContext<CanvasSessionContextValue | null>(null);

export const CanvasSessionContextProvider = memo(
  ({ id, type, children }: PropsWithChildren<{ id: string; type: 'simple' | 'advanced' }>) => {
    /**
     * For best performance and interop with the Canvas, which is outside react but needs to interact with the react
     * app, all canvas session state is packaged as nanostores atoms. The trickiest part is syncing the queue items
     * with a nanostores atom.
     */
    const session = useMemo(() => ({ type, id }), [type, id]);

    /**
     * App store
     */
    const store = useAppStore();

    const socket = useStore($socket);

    /**
     * Manually-synced atom containing queue items for the current session. This is populated from the RTK Query cache
     * and kept in sync with it via a redux subscription.
     */
    const $items = useState(() => atom<S['SessionQueueItem'][]>([]))[0];

    /**
     * Whether auto-switch is enabled.
     */
    const $autoSwitch = useState(() => atom<AutoSwitchMode>('first_progress'))[0];

    /**
     * An internal flag used to work around race conditions with auto-switch switching to queue items before their
     * output images have fully loaded.
     */
    const $lastLoadedItemId = useState(() => atom<number | null>(null))[0];

    /**
     * An ephemeral store of progress events and images for all items in the current session.
     */
    const $progressData = useState(() => atom<Record<number, ProgressData>>({}))[0];

    /**
     * The currently selected queue item's ID, or null if one is not selected.
     */
    const $selectedItemId = useState(() => atom<number | null>(null))[0];

    /**
     * The number of items. Computed from the queue items array.
     */
    const $itemCount = useState(() => computed([$items], (items) => items.length))[0];

    /**
     * Whether there are any items. Computed from the queue items array.
     */
    const $hasItems = useState(() => computed([$items], (items) => items.length > 0))[0];

    /**
     * The currently selected queue item, or null if one is not selected.
     */
    const $selectedItem = useState(() =>
      computed([$items, $selectedItemId], (items, selectedItemId) => {
        if (items.length === 0) {
          return null;
        }
        if (selectedItemId === null) {
          return null;
        }
        return items.find(({ item_id }) => item_id === selectedItemId) ?? null;
      })
    )[0];

    /**
     * The currently selected queue item's index in the list of items, or null if one is not selected.
     */
    const $selectedItemIndex = useState(() =>
      computed([$items, $selectedItemId], (items, selectedItemId) => {
        if (items.length === 0) {
          return null;
        }
        if (selectedItemId === null) {
          return null;
        }
        return items.findIndex(({ item_id }) => item_id === selectedItemId) ?? null;
      })
    )[0];

    /**
     * The currently selected queue item's output image name, or null if one is not selected or there is no output
     * image recorded.
     */
    const $selectedItemOutputImageName = useState(() =>
      computed([$selectedItemId, $progressData], (selectedItemId, progressData) => {
        if (selectedItemId === null) {
          return null;
        }
        const datum = progressData[selectedItemId];
        if (!datum) {
          return null;
        }
        return datum.outputImageName;
      })
    )[0];

    /**
     * A redux selector to select all queue items from the RTK Query cache. It's important that this returns stable
     * references if possible to reduce re-renders. All derivations of the queue items (e.g. filtering out canceled
     * items) should be done in a nanostores computed.
     */
    const selectQueueItems = useMemo(
      () =>
        createSelector(
          queueApi.endpoints.listAllQueueItems.select({ destination: session.id }),
          ({ data }) => data ?? EMPTY_ARRAY
        ),
      [session.id]
    );

    const selectNext = useCallback(() => {
      const selectedItemId = $selectedItemId.get();
      if (selectedItemId === null) {
        return;
      }
      const items = $items.get();
      const currentIndex = items.findIndex((item) => item.item_id === selectedItemId);
      const nextIndex = (currentIndex + 1) % items.length;
      const nextItem = items[nextIndex];
      if (!nextItem) {
        return;
      }
      $selectedItemId.set(nextItem.item_id);
    }, [$items, $selectedItemId]);

    const selectPrev = useCallback(() => {
      const selectedItemId = $selectedItemId.get();
      if (selectedItemId === null) {
        return;
      }
      const items = $items.get();
      const currentIndex = items.findIndex((item) => item.item_id === selectedItemId);
      const prevIndex = (currentIndex - 1 + items.length) % items.length;
      const prevItem = items[prevIndex];
      if (!prevItem) {
        return;
      }
      $selectedItemId.set(prevItem.item_id);
    }, [$items, $selectedItemId]);

    const selectFirst = useCallback(() => {
      const items = $items.get();
      const first = items.at(0);
      if (!first) {
        return;
      }
      $selectedItemId.set(first.item_id);
    }, [$items, $selectedItemId]);

    const selectLast = useCallback(() => {
      const items = $items.get();
      const last = items.at(-1);
      if (!last) {
        return;
      }
      $selectedItemId.set(last.item_id);
    }, [$items, $selectedItemId]);

    // Set up socket listeners
    useEffect(() => {
      if (!socket) {
        return;
      }

      const onProgress = (data: S['InvocationProgressEvent']) => {
        if (data.destination !== session.id) {
          return;
        }
        const isFirstProgressImage = !$progressData.get()[data.item_id]?.progressImage && !!data.image;
        setProgress($progressData, data);
        if ($autoSwitch.get() === 'first_progress' && isFirstProgressImage) {
          $selectedItemId.set(data.item_id);
        }
      };

      socket.on('invocation_progress', onProgress);

      return () => {
        socket.off('invocation_progress', onProgress);
      };
    }, [$autoSwitch, $progressData, $selectedItemId, session.id, socket]);

    // Set up state subscriptions and effects
    useEffect(() => {
      let _prevItems: readonly S['SessionQueueItem'][] = [];
      // Seed the $items atom with the initial query cache state
      $items.set(selectQueueItems(store.getState()));

      // Manually keep the $items atom in sync as the query cache is updated
      const unsubReduxSyncToItemsAtom = store.subscribe(() => {
        const prevItems = $items.get();
        const items = selectQueueItems(store.getState());
        if (items !== prevItems) {
          _prevItems = prevItems;
          $items.set(items);
        }
      });

      // Handle cases that could result in a nonexistent queue item being selected.
      const unsubEnsureSelectedItemIdExists = effect([$items, $selectedItemId], (items, selectedItemId) => {
        // If there are no items, cannot have a selected item.
        if (items.length === 0) {
          $selectedItemId.set(null);
          return;
        }
        // If there is no selected item but there are items, select the first one.
        if (selectedItemId === null && items.length > 0) {
          $selectedItemId.set(items[0]?.item_id ?? null);
          return;
        }
        // If an item is selected and it is not in the list of items, un-set it. This effect will run again and we'll
        // the above case, selecting the first item if there are any.
        if (selectedItemId !== null && items.findIndex(({ item_id }) => item_id === selectedItemId) === -1) {
          let prevIndex = _prevItems.findIndex(({ item_id }) => item_id === selectedItemId);
          if (prevIndex >= items.length) {
            prevIndex = items.length - 1;
          }
          const nextItem = items[prevIndex];
          $selectedItemId.set(nextItem?.item_id ?? null);
          return;
        }
      });

      // Clean up the progress data when a queue item is discarded.
      const unsubCleanUpProgressData = $items.listen((items) => {
        const progressData = $progressData.get();

        const toDelete: number[] = [];
        const toUpdate: ProgressData[] = [];

        for (const datum of Object.values(progressData)) {
          const item = items.find(({ item_id }) => item_id === datum.itemId);
          if (!item) {
            toDelete.push(datum.itemId);
          } else if (item.status === 'canceled' || item.status === 'failed') {
            toUpdate[datum.itemId] = {
              ...datum,
              progressEvent: null,
              progressImage: null,
              outputImageName: null,
            };
          }
        }

        for (const item of items) {
          const datum = progressData[item.item_id];

          if (datum) {
            if (datum.outputImageName) {
              continue;
            }
            const outputImageName = getOutputImageName(item);
            if (!outputImageName) {
              continue;
            }
            toUpdate.push({
              ...datum,
              outputImageName,
            });
          } else {
            const _datum = getInitialProgressData(item.item_id);
            _datum.outputImageName = getOutputImageName(item);
            toUpdate.push(_datum);
          }
        }

        if (toDelete.length === 0 && toUpdate.length === 0) {
          return;
        }

        const newProgressData = { ...progressData };

        for (const itemId of toDelete) {
          delete newProgressData[itemId];
        }

        for (const datum of toUpdate) {
          newProgressData[datum.itemId] = datum;
        }

        $progressData.set(newProgressData);
      });

      // We only want to auto-switch to completed queue items once their images have fully loaded to prevent flashes
      // of fallback content and/or progress images. The only surefire way to determine when images have fully loaded
      // is via the image elements' `onLoad` callback. Images set `$lastLoadedItemId` to their queue item ID in their
      // `onLoad` handler, and we listen for that here. If auto-switch is enabled, we then switch the to the item.
      //
      // TODO: This isn't perfect... we set $lastLoadedItemId in the mini preview component, but the full view
      // component still needs to retrieve the image from the browser cache... can result in a flash of the progress
      // image...
      const unsubHandleAutoSwitch = $lastLoadedItemId.listen((lastLoadedItemId) => {
        if (lastLoadedItemId === null) {
          return;
        }
        if ($autoSwitch.get() === 'completed') {
          $selectedItemId.set(lastLoadedItemId);
        }
        $lastLoadedItemId.set(null);
      });

      // Create an RTK Query subscription. Without this, the query cache selector will never return anything bc RTK
      // doesn't know we care about it.
      const { unsubscribe: unsubQueueItemsQuery } = store.dispatch(
        queueApi.endpoints.listAllQueueItems.initiate({ destination: session.id })
      );

      // Clean up all subscriptions and top-level (i.e. non-computed/derived state)
      return () => {
        unsubHandleAutoSwitch();
        unsubQueueItemsQuery();
        unsubReduxSyncToItemsAtom();
        unsubEnsureSelectedItemIdExists();
        unsubCleanUpProgressData();
        $items.set([]);
        $progressData.set({});
        $selectedItemId.set(null);
      };
    }, [$autoSwitch, $items, $lastLoadedItemId, $progressData, $selectedItemId, selectQueueItems, session.id, store]);

    const value = useMemo<CanvasSessionContextValue>(
      () => ({
        session,
        $items,
        $hasItems,
        $progressData,
        $selectedItemId,
        $autoSwitch,
        $selectedItem,
        $selectedItemIndex,
        $lastLoadedItemId,
        $selectedItemOutputImageName,
        $itemCount,
        selectNext,
        selectPrev,
        selectFirst,
        selectLast,
      }),
      [
        $autoSwitch,
        $items,
        $hasItems,
        $lastLoadedItemId,
        $progressData,
        $selectedItem,
        $selectedItemId,
        $selectedItemIndex,
        session,
        $selectedItemOutputImageName,
        $itemCount,
        selectNext,
        selectPrev,
        selectFirst,
        selectLast,
      ]
    );

    return <CanvasSessionContext.Provider value={value}>{children}</CanvasSessionContext.Provider>;
  }
);
CanvasSessionContextProvider.displayName = 'CanvasSessionContextProvider';

export const useCanvasSessionContext = () => {
  const ctx = useContext(CanvasSessionContext);
  assert(ctx !== null, "'useCanvasSessionContext' must be used within a CanvasSessionContextProvider");
  return ctx;
};
