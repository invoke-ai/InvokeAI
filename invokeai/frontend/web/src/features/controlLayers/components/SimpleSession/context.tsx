import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppStore } from 'app/store/nanostores/store';
import type {
  AdvancedSessionIdentifier,
  SimpleSessionIdentifier,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import type { ProgressImage } from 'features/nodes/types/common';
import type { Atom, WritableAtom } from 'nanostores';
import { atom, computed, effect } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useEffect, useMemo, useState } from 'react';
import { queueApi } from 'services/api/endpoints/queue';
import type { S } from 'services/api/types';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';

export type ProgressData = {
  sessionId: string;
  progressEvent: S['InvocationProgressEvent'] | null;
  progressImage: ProgressImage | null;
};

export const buildProgressDataAtom = () => atom<Record<string, ProgressData>>({});

export const useProgressData = (
  $progressData: WritableAtom<Record<string, ProgressData>>,
  sessionId: string
): ProgressData => {
  const [value, setValue] = useState<ProgressData>(() => {
    return $progressData.get()[sessionId] ?? { sessionId, progressEvent: null, progressImage: null };
  });
  useEffect(() => {
    const unsub = $progressData.subscribe((data) => {
      const progressData = data[sessionId];
      if (!progressData) {
        return;
      }
      setValue(progressData);
    });
    return () => {
      unsub();
    };
  }, [$progressData, sessionId]);

  return value;
};

export const useHasProgressImage = (
  $progressData: WritableAtom<Record<string, ProgressData>>,
  sessionId: string
): boolean => {
  const [value, setValue] = useState(false);
  useEffect(() => {
    const unsub = $progressData.subscribe((data) => {
      const progressData = data[sessionId];
      setValue(Boolean(progressData?.progressImage));
    });
    return () => {
      unsub();
    };
  }, [$progressData, sessionId]);

  return value;
};

export const setProgress = (
  $progressData: WritableAtom<Record<string, ProgressData>>,
  data: S['InvocationProgressEvent']
) => {
  const progressData = $progressData.get();
  const current = progressData[data.session_id];
  if (current) {
    const next = { ...current };
    next.progressEvent = data;
    if (data.image) {
      next.progressImage = data.image;
    }
    $progressData.set({
      ...progressData,
      [data.session_id]: next,
    });
  } else {
    $progressData.set({
      ...progressData,
      [data.session_id]: {
        sessionId: data.session_id,
        progressEvent: data,
        progressImage: data.image ?? null,
      },
    });
  }
};

export const clearProgressEvent = ($progressData: WritableAtom<Record<string, ProgressData>>, sessionId: string) => {
  const progressData = $progressData.get();
  const current = progressData[sessionId];
  if (!current) {
    return;
  }
  const next = { ...current };
  next.progressEvent = null;
  $progressData.set({
    ...progressData,
    [sessionId]: next,
  });
};

export const clearProgressImage = ($progressData: WritableAtom<Record<string, ProgressData>>, sessionId: string) => {
  const progressData = $progressData.get();
  const current = progressData[sessionId];
  if (!current) {
    return;
  }
  const next = { ...current };
  next.progressImage = null;
  $progressData.set({
    ...progressData,
    [sessionId]: next,
  });
};

export type CanvasSessionContextValue = {
  session: SimpleSessionIdentifier | AdvancedSessionIdentifier;
  $items: Atom<S['SessionQueueItem'][]>;
  $hasItems: Atom<boolean>;
  $progressData: WritableAtom<Record<string, ProgressData>>;
  $selectedItemId: WritableAtom<number | null>;
  $selectedItem: Atom<S['SessionQueueItem'] | null>;
  $selectedItemIndex: Atom<number | null>;
  $autoSwitch: WritableAtom<boolean>;
};

const CanvasSessionContext = createContext<CanvasSessionContextValue | null>(null);

export const CanvasSessionContextProvider = memo(
  ({ session, children }: PropsWithChildren<{ session: SimpleSessionIdentifier | AdvancedSessionIdentifier }>) => {
    /**
     * For best performance and interop with the Canvas, which is outside react but needs to interact with the react
     * app, all canvas session state is packaged as nanostores atoms. The trickiest part is syncing the queue items
     * with a nanostores atom.
     */

    /**
     * App store
     */
    const store = useAppStore();

    const socket = useStore($socket);

    /**
     * Manually-synced atom containing the queue items for the current session.
     */
    const $items = useState(() => atom<S['SessionQueueItem'][]>([]))[0];

    /**
     * Whether auto-switch is enabled.
     */
    const $autoSwitch = useState(() => atom(true))[0];

    /**
     * An ephemeral store of progress events and images for all items in the current session.
     */
    const $progressData = useState(() => atom<Record<string, ProgressData>>({}))[0];

    /**
     * The currently selected queue item's ID, or null if one is not selected.
     */
    const $selectedItemId = useState(() => atom<number | null>(null))[0];

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

    // Set up socket listeners
    useEffect(() => {
      if (!socket) {
        return;
      }

      const onProgress = (data: S['InvocationProgressEvent']) => {
        if (data.destination !== session.id) {
          return;
        }
        setProgress($progressData, data);
      };

      const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
        if (data.destination !== session.id) {
          return;
        }
        if (data.status === 'completed' && $autoSwitch.get()) {
          $selectedItemId.set(data.item_id);
        }
      };

      socket.on('invocation_progress', onProgress);
      socket.on('queue_item_status_changed', onQueueItemStatusChanged);

      return () => {
        socket.off('invocation_progress', onProgress);
        socket.off('queue_item_status_changed', onQueueItemStatusChanged);
      };
    }, [$autoSwitch, $progressData, $selectedItemId, session.id, socket]);

    // Set up state subscriptions and effects
    useEffect(() => {
      // Seed the $items atom with the initial query cache state
      $items.set(selectQueueItems(store.getState()));

      // Manually keep the $items atom in sync as the query cache is updated
      const unsubReduxSyncToItemsAtom = store.subscribe(() => {
        const prevItems = $items.get();
        const items = selectQueueItems(store.getState());
        if (items !== prevItems) {
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
          $selectedItemId.set(null);
          return;
        }
      });

      // Clean up the progress data when a queue item is discarded.
      const unsubCleanUpProgressData = effect([$items, $progressData], (items, progressData) => {
        const toDelete: string[] = [];
        for (const datum of Object.values(progressData)) {
          if (items.findIndex(({ session_id }) => session_id === datum.sessionId) === -1) {
            toDelete.push(datum.sessionId);
          }
        }
        if (toDelete.length === 0) {
          return;
        }
        const newProgressData = { ...progressData };
        for (const sessionId of toDelete) {
          delete newProgressData[sessionId];
        }
        // This will re-trigger the effect - maybe this could just be a listener on $items? Brain hurt
        $progressData.set(newProgressData);
      });

      // Create an RTK Query subscription. Without this, the query cache selector will never return anything bc RTK
      // doesn't know we care about it.
      const { unsubscribe: unsubQueueItemsQuery } = store.dispatch(
        queueApi.endpoints.listAllQueueItems.initiate({ destination: session.id })
      );

      // Clean up all subscriptions and top-level (i.e. non-computed/derived state)
      return () => {
        unsubQueueItemsQuery();
        unsubReduxSyncToItemsAtom();
        unsubEnsureSelectedItemIdExists();
        unsubCleanUpProgressData();
        $items.set([]);
        $progressData.set({});
        $selectedItemId.set(null);
      };
    }, [$items, $progressData, $selectedItemId, selectQueueItems, session.id, store]);

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
      }),
      [$autoSwitch, $hasItems, $items, $progressData, $selectedItem, $selectedItemId, $selectedItemIndex, session]
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
