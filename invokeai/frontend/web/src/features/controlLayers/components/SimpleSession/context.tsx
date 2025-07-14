import { useStore } from '@nanostores/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { getOutputImageName } from 'features/controlLayers/components/SimpleSession/shared';
import { loadImage } from 'features/controlLayers/konva/util';
import { selectStagingAreaAutoSwitch } from 'features/controlLayers/store/canvasSettingsSlice';
import {
  buildSelectCanvasQueueItems,
  canvasQueueItemDiscarded,
  canvasSessionReset,
  selectCanvasSessionId,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import type { ProgressImage } from 'features/nodes/types/common';
import type { Atom, MapStore, StoreValue, WritableAtom } from 'nanostores';
import { atom, computed, effect, map, subscribeKeys } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { getImageDTOSafe } from 'services/api/endpoints/images';
import { queueApi } from 'services/api/endpoints/queue';
import type { ImageDTO, S } from 'services/api/types';
import { $socket } from 'services/events/stores';
import { assert, objectEntries } from 'tsafe';

export type ProgressData = {
  itemId: number;
  progressEvent: S['InvocationProgressEvent'] | null;
  progressImage: ProgressImage | null;
  imageDTO: ImageDTO | null;
  imageLoaded: boolean;
};

const getInitialProgressData = (itemId: number): ProgressData => ({
  itemId,
  progressEvent: null,
  progressImage: null,
  imageDTO: null,
  imageLoaded: false,
});

export const useProgressData = ($progressData: ProgressDataMap, itemId: number): ProgressData => {
  const getInitialValue = useCallback(
    () => $progressData.get()[itemId] ?? getInitialProgressData(itemId),
    [$progressData, itemId]
  );
  const [value, setValue] = useState(getInitialValue);
  useEffect(() => {
    const unsub = subscribeKeys($progressData, [itemId], (data) => {
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

const setProgress = ($progressData: ProgressDataMap, data: S['InvocationProgressEvent']) => {
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
        imageDTO: null,
        imageLoaded: false,
      },
    });
  }
};

export type ProgressDataMap = MapStore<Record<number, ProgressData | undefined>>;

type CanvasSessionContextValue = {
  $items: Atom<S['SessionQueueItem'][]>;
  $itemCount: Atom<number>;
  $hasItems: Atom<boolean>;
  $progressData: ProgressDataMap;
  $selectedItemId: WritableAtom<number | null>;
  $selectedItem: Atom<S['SessionQueueItem'] | null>;
  $selectedItemIndex: Atom<number | null>;
  $selectedItemOutputImageDTO: Atom<ImageDTO | null>;
  selectNext: () => void;
  selectPrev: () => void;
  selectFirst: () => void;
  selectLast: () => void;
  discard: (itemId: number) => void;
  discardAll: () => void;
};

const CanvasSessionContext = createContext<CanvasSessionContextValue | null>(null);

export const CanvasSessionContextProvider = memo(({ children }: PropsWithChildren) => {
  /**
   * For best performance and interop with the Canvas, which is outside react but needs to interact with the react
   * app, all canvas session state is packaged as nanostores atoms. The trickiest part is syncing the queue items
   * with a nanostores atom.
   */

  /**
   * App store
   */
  const store = useAppStore();

  const sessionId = useAppSelector(selectCanvasSessionId);

  const socket = useStore($socket);

  /**
   * Track the last completed item. Used to implement autoswitch.
   */
  const $lastCompletedItemId = useState(() => atom<number | null>(null))[0];

  /**
   * Manually-synced atom containing queue items for the current session. This is populated from the RTK Query cache
   * and kept in sync with it via a redux subscription.
   */
  const $items = useState(() => atom<S['SessionQueueItem'][]>([]))[0];

  /**
   * An ephemeral store of progress events and images for all items in the current session.
   */
  const $progressData = useState(() => map<StoreValue<ProgressDataMap>>({}))[0];

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
   * Whether there are any pending or in-progress items. Computed from the queue items array.
   */
  const $isPending = useState(() =>
    computed([$items], (items) => items.some((item) => item.status === 'pending' || item.status === 'in_progress'))
  )[0];

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
  const $selectedItemOutputImageDTO = useState(() =>
    computed([$selectedItemId, $progressData], (selectedItemId, progressData) => {
      if (selectedItemId === null) {
        return null;
      }
      const datum = progressData[selectedItemId];
      if (!datum) {
        return null;
      }
      return datum.imageDTO;
    })
  )[0];

  /**
   * A redux selector to select all queue items from the RTK Query cache.
   */
  const selectQueueItems = useMemo(() => buildSelectCanvasQueueItems(sessionId), [sessionId]);

  const discard = useCallback(
    (itemId: number) => {
      store.dispatch(canvasQueueItemDiscarded({ itemId }));
    },
    [store]
  );

  const discardAll = useCallback(() => {
    store.dispatch(canvasSessionReset());
  }, [store]);

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
      if (data.destination !== sessionId) {
        return;
      }
      setProgress($progressData, data);
    };

    const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
      if (data.destination !== sessionId) {
        return;
      }
      if (data.status === 'completed') {
        /**
         * There is an unpleasant bit of indirection here. When an item is completed, and auto-switch is set to
         * switch_on_finish, we want to load the image and switch to it. In this socket handler, we don't have
         * access to the full queue item, which we need to get the output image and load it. We get the full
         * queue items as part of the list query, so it's rather inefficient to fetch it again here.
         *
         * To reduce the number of extra network requests, we instead store this item as the last completed item.
         * Then in the progress data sync effect, we process the queue item load its image.
         */
        $lastCompletedItemId.set(data.item_id);
      }
      if (data.status === 'in_progress' && selectStagingAreaAutoSwitch(store.getState()) === 'switch_on_start') {
        $selectedItemId.set(data.item_id);
      }
    };

    socket.on('invocation_progress', onProgress);
    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      socket.off('invocation_progress', onProgress);
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [$progressData, $selectedItemId, sessionId, socket, $lastCompletedItemId, store]);

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
      if (items.length === 0) {
        // If there are no items, cannot have a selected item.
        $selectedItemId.set(null);
      } else if (selectedItemId === null && items.length > 0) {
        // If there is no selected item but there are items, select the first one.
        $selectedItemId.set(items[0]?.item_id ?? null);
        return;
      } else if (selectedItemId !== null && items.findIndex(({ item_id }) => item_id === selectedItemId) === -1) {
        // If an item is selected and it is not in the list of items, un-set it. This effect will run again and we'll
        // the above case, selecting the first item if there are any.
        let prevIndex = _prevItems.findIndex(({ item_id }) => item_id === selectedItemId);
        if (prevIndex >= items.length) {
          prevIndex = items.length - 1;
        }
        const nextItem = items[prevIndex];
        $selectedItemId.set(nextItem?.item_id ?? null);
      }

      if (items !== _prevItems) {
        _prevItems = items;
      }
    });

    // Sync progress data - remove canceled/failed items, update progress data with new images, and load images
    const unsubSyncProgressData = $items.subscribe(async (items) => {
      const progressData = $progressData.get();

      const toDelete: number[] = [];
      const toUpdate: ProgressData[] = [];

      for (const [id, datum] of objectEntries(progressData)) {
        if (!datum) {
          toDelete.push(id);
          continue;
        }
        const item = items.find(({ item_id }) => item_id === datum.itemId);
        if (!item) {
          toDelete.push(datum.itemId);
        } else if (item.status === 'canceled' || item.status === 'failed') {
          toUpdate.push({
            ...datum,
            progressEvent: null,
            progressImage: null,
            imageDTO: null,
          });
        }
      }

      for (const item of items) {
        const datum = progressData[item.item_id];

        if (datum?.imageDTO) {
          continue;
        }
        const outputImageName = getOutputImageName(item);
        if (!outputImageName) {
          continue;
        }
        const imageDTO = await getImageDTOSafe(outputImageName);
        if (!imageDTO) {
          continue;
        }

        // This is the load logic mentioned in the comment in the QueueItemStatusChangedEvent handler above.
        if (
          $lastCompletedItemId.get() === item.item_id &&
          selectStagingAreaAutoSwitch(store.getState()) === 'switch_on_finish'
        ) {
          loadImage(imageDTO.image_url, true).then(() => {
            $selectedItemId.set(item.item_id);
            $lastCompletedItemId.set(null);
          });
        }

        toUpdate.push({
          ...getInitialProgressData(item.item_id),
          ...datum,
          imageDTO,
        });
      }

      for (const itemId of toDelete) {
        $progressData.setKey(itemId, undefined);
      }

      for (const datum of toUpdate) {
        $progressData.setKey(datum.itemId, datum);
      }
    });

    // Create an RTK Query subscription. Without this, the query cache selector will never return anything bc RTK
    // doesn't know we care about it.
    const { unsubscribe: unsubQueueItemsQuery } = store.dispatch(
      queueApi.endpoints.listAllQueueItems.initiate({ destination: sessionId })
    );

    // Clean up all subscriptions and top-level (i.e. non-computed/derived state)
    return () => {
      unsubQueueItemsQuery();
      unsubReduxSyncToItemsAtom();
      unsubEnsureSelectedItemIdExists();
      unsubSyncProgressData();
      $items.set([]);
      $progressData.set({});
      $selectedItemId.set(null);
    };
  }, [$items, $progressData, $selectedItemId, selectQueueItems, sessionId, store, $lastCompletedItemId]);

  const value = useMemo<CanvasSessionContextValue>(
    () => ({
      $items,
      $hasItems,
      $isPending,
      $progressData,
      $selectedItemId,
      $selectedItem,
      $selectedItemIndex,
      $selectedItemOutputImageDTO,
      $itemCount,
      selectNext,
      selectPrev,
      selectFirst,
      selectLast,
      discard,
      discardAll,
    }),
    [
      $items,
      $hasItems,
      $isPending,
      $progressData,
      $selectedItem,
      $selectedItemId,
      $selectedItemIndex,
      $selectedItemOutputImageDTO,
      $itemCount,
      selectNext,
      selectPrev,
      selectFirst,
      selectLast,
      discard,
      discardAll,
    ]
  );

  return <CanvasSessionContext.Provider value={value}>{children}</CanvasSessionContext.Provider>;
});
CanvasSessionContextProvider.displayName = 'CanvasSessionContextProvider';

export const useCanvasSessionContext = () => {
  const ctx = useContext(CanvasSessionContext);
  assert(ctx !== null, "'useCanvasSessionContext' must be used within a CanvasSessionContextProvider");
  return ctx;
};

export const useOutputImageDTO = (item: S['SessionQueueItem']) => {
  const ctx = useCanvasSessionContext();
  const $imageDTO = useState(() =>
    computed([ctx.$progressData], (progressData) => progressData[item.item_id]?.imageDTO ?? null)
  )[0];
  const imageDTO = useStore($imageDTO);

  return imageDTO;
};
