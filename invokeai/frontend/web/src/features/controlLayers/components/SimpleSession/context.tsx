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
    const store = useAppStore();
    const [$items] = useState(() => atom<S['SessionQueueItem'][]>([]));
    const [$hasItems] = useState(() => computed([$items], (items) => items.length > 0));
    const [$autoSwitch] = useState(() => atom(true));
    const [$selectedItemId] = useState(() => atom<number | null>(null));
    const [$progressData] = useState(() => atom<Record<string, ProgressData>>({}));
    const [$selectedItem] = useState(() =>
      computed([$items, $selectedItemId], (items, selectedItemId) => {
        if (items.length === 0) {
          return null;
        }
        if (selectedItemId === null) {
          return null;
        }
        return items.find(({ item_id }) => item_id === selectedItemId) ?? null;
      })
    );
    const [$selectedItemIndex] = useState(() =>
      computed([$items, $selectedItemId], (items, selectedItemId) => {
        if (items.length === 0) {
          return null;
        }
        if (selectedItemId === null) {
          return null;
        }
        return items.findIndex(({ item_id }) => item_id === selectedItemId) ?? null;
      })
    );

    const selectQueueItems = useMemo(
      () =>
        createSelector(
          queueApi.endpoints.listAllQueueItems.select({ destination: session.id }),
          ({ data }) => data?.filter((item) => item.status !== 'canceled') ?? EMPTY_ARRAY
        ),
      [session.id]
    );

    useEffect(() => {
      $items.set(selectQueueItems(store.getState()));

      const unsubReduxSyncToItemsAtom = store.subscribe(() => {
        const prevItems = $items.get();
        const items = selectQueueItems(store.getState());
        if (items !== prevItems) {
          $items.set(items);
        }
      });

      const unsubEnsureSelectedItemIdExists = effect([$items, $selectedItemId], (items, selectedItemId) => {
        if (items.length === 0) {
          $selectedItemId.set(null);
          return;
        }
        if (selectedItemId === null && items.length > 0) {
          $selectedItemId.set(items[0]?.item_id ?? null);
          return;
        }
        if (selectedItemId !== null && items.findIndex(({ item_id }) => item_id === selectedItemId) === -1) {
          $selectedItemId.set(null);
          return;
        }
      });

      const { unsubscribe: unsubQueueItemsQuery } = store.dispatch(
        queueApi.endpoints.listAllQueueItems.initiate({ destination: session.id })
      );

      return () => {
        unsubQueueItemsQuery();
        unsubReduxSyncToItemsAtom();
        unsubEnsureSelectedItemIdExists();
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
