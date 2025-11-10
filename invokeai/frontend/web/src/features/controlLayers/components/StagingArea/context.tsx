import { useStore } from '@nanostores/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import {
  selectStagingAreaAutoSwitch,
  settingsStagingAreaAutoSwitchChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import {
  canvasQueueItemDiscarded,
  canvasSessionReset,
  selectActiveCanvasQueueItems,
  selectActiveCanvasStagingAreaSessionId,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectBboxRect, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useEffect, useMemo, useState } from 'react';
import { getImageDTOSafe } from 'services/api/endpoints/images';
import { queueApi } from 'services/api/endpoints/queue';
import type { S } from 'services/api/types';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';

import type { ProgressData, StagingAreaAppApi } from './state';
import { getInitialProgressData, StagingAreaApi } from './state';

const StagingAreaContext = createContext<StagingAreaApi | null>(null);

export const StagingAreaContextProvider = memo(({ children }: PropsWithChildren) => {
  const store = useAppStore();
  const socket = useStore($socket);
  const sessionId = useAppSelector(selectActiveCanvasStagingAreaSessionId);

  const stagingAreaAppApi = useMemo<StagingAreaAppApi>(() => {
    const _stagingAreaAppApi: StagingAreaAppApi = {
      getAutoSwitch: () => {
        return selectStagingAreaAutoSwitch(store.getState());
      },
      getImageDTO: (imageName: string) => getImageDTOSafe(imageName),
      onInvocationProgress: (handler) => {
        socket?.on('invocation_progress', handler);
        return () => {
          socket?.off('invocation_progress', handler);
        };
      },
      onQueueItemStatusChanged: (handler) => {
        socket?.on('queue_item_status_changed', handler);
        return () => {
          socket?.off('queue_item_status_changed', handler);
        };
      },
      onItemsChanged: (handler) => {
        let prev: S['SessionQueueItem'][] = [];
        return store.subscribe(() => {
          const next = selectActiveCanvasQueueItems(store.getState());
          if (prev !== next) {
            prev = next;
            handler(next);
          }
        });
      },
      onDiscard: ({ item_id, status }) => {
        store.dispatch(canvasQueueItemDiscarded({ itemId: item_id }));
        if (status === 'in_progress' || status === 'pending') {
          store.dispatch(queueApi.endpoints.cancelQueueItem.initiate({ item_id }, { track: false }));
        }
      },
      onDiscardAll: () => {
        store.dispatch(canvasSessionReset());
        if (sessionId) {
          store.dispatch(
            queueApi.endpoints.cancelQueueItemsByDestination.initiate({ destination: sessionId }, { track: false })
          );
        }
      },
      onAccept: (item, imageDTO) => {
        const bboxRect = selectBboxRect(store.getState());
        const { x, y } = bboxRect;
        const imageObject = imageDTOToImageObject(imageDTO);
        const selectedEntityIdentifier = selectSelectedEntityIdentifier(store.getState());
        const overrides: Partial<CanvasRasterLayerState> = {
          position: { x, y },
          objects: [imageObject],
        };

        store.dispatch(rasterLayerAdded({ overrides, isSelected: selectedEntityIdentifier?.type === 'raster_layer' }));
        store.dispatch(canvasSessionReset());
        if (sessionId) {
          store.dispatch(
            queueApi.endpoints.cancelQueueItemsByDestination.initiate({ destination: sessionId }, { track: false })
          );
        }
      },
      onAutoSwitchChange: (mode) => {
        store.dispatch(settingsStagingAreaAutoSwitchChanged(mode));
      },
    };

    return _stagingAreaAppApi;
  }, [sessionId, socket, store]);

  const [stagingAreaApi] = useState(() => new StagingAreaApi());

  useEffect(() => {
    if (!sessionId) {
      return () => {
        stagingAreaApi.cleanup();
      };
    }

    stagingAreaApi.connectToApp(sessionId, stagingAreaAppApi);

    // We need to subscribe to the queue items query manually to ensure the staging area actually gets the items
    const { unsubscribe: unsubQueueItemsQuery } = store.dispatch(
      queueApi.endpoints.listAllQueueItems.initiate({ destination: sessionId })
    );

    return () => {
      stagingAreaApi.cleanup();
      unsubQueueItemsQuery();
    };
  }, [sessionId, stagingAreaApi, stagingAreaAppApi, store]);

  return <StagingAreaContext.Provider value={stagingAreaApi}>{children}</StagingAreaContext.Provider>;
});
StagingAreaContextProvider.displayName = 'StagingAreaContextProvider';

export const useStagingAreaContext = () => {
  const ctx = useContext(StagingAreaContext);
  assert(ctx !== null, "'useStagingAreaContext' must be used within a StagingAreaContextProvider");
  return ctx;
};

export const useOutputImageDTO = (itemId: number) => {
  const ctx = useStagingAreaContext();
  const allProgressData = useStore(ctx.$progressData, { keys: [itemId] });
  return allProgressData[itemId]?.imageDTO ?? null;
};

export const useProgressDatum = (itemId: number): ProgressData => {
  const ctx = useStagingAreaContext();
  const allProgressData = useStore(ctx.$progressData, { keys: [itemId] });
  return allProgressData[itemId] ?? getInitialProgressData(itemId);
};
