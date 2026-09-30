import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppStore } from 'app/store/storeHooks';
import { canvasSnapshotRestored } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { zCanvasState } from 'features/controlLayers/store/types';
import { collectImageNames } from 'features/controlLayers/util/canvasProjectFile';
import { useCallback, useMemo } from 'react';
import { serializeError } from 'serialize-error';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import {
  clientStateApi,
  useDeleteClientStateByKeyMutation,
  useGetClientStateKeysByPrefixQuery,
  useSetClientStateByKeyMutation,
} from 'services/api/endpoints/clientState';
import { getImageDTOSafe } from 'services/api/endpoints/images';
import type { JsonObject } from 'type-fest';
import { z } from 'zod';

const log = logger('canvas');

const SNAPSHOT_PREFIX = 'canvas_snapshot:';

/**
 * Quick health check to determine if the backend is reachable.
 * Uses the existing appInfoApi RTKQ endpoint for consistency.
 */
const isBackendReachable = async (dispatch: ReturnType<typeof useAppDispatch>): Promise<boolean> => {
  const req = dispatch(appInfoApi.endpoints.getAppVersion.initiate(undefined, { subscribe: false }));
  try {
    await req.unwrap();
    return true;
  } catch {
    return false;
  } finally {
    req.unsubscribe();
  }
};

/**
 * Check which image_names still exist on the server.
 * Returns the list of missing image names. If the backend is unreachable,
 * skips all checks and returns an empty array to avoid false warnings.
 */
const findMissingImages = async (
  imageNames: string[],
  dispatch: ReturnType<typeof useAppDispatch>
): Promise<string[]> => {
  // Pre-flight: verify backend is reachable before checking individual images
  if (!(await isBackendReachable(dispatch))) {
    log.warn('Backend unreachable — skipping missing image check');
    return [];
  }

  const results = await Promise.all(
    imageNames.map(async (name) => {
      const dto = await getImageDTOSafe(name);
      return dto === null ? name : null;
    })
  );
  return results.filter((name): name is string => name !== null);
};

export type SnapshotInfo = {
  key: string;
  name: string;
};

type RestoreResult = {
  success: boolean;
  missingImageCount?: number;
  error?: 'incompatible' | 'not_found' | 'unknown';
};

export const useCanvasSnapshots = () => {
  const dispatch = useAppDispatch();
  const store = useAppStore();

  const { data: keys } = useGetClientStateKeysByPrefixQuery(SNAPSHOT_PREFIX);
  const [setClientState] = useSetClientStateByKeyMutation();
  const [deleteClientState] = useDeleteClientStateByKeyMutation();

  const snapshots: SnapshotInfo[] = useMemo(
    () =>
      (keys ?? []).map((key) => ({
        key,
        name: key.slice(SNAPSHOT_PREFIX.length),
      })),
    [keys]
  );

  const saveSnapshot = useCallback(
    async (name: string) => {
      try {
        const state = selectCanvasSlice(store.getState());
        const value = JSON.stringify(state);
        const key = `${SNAPSHOT_PREFIX}${name}`;
        await setClientState({ key, value }).unwrap();
        return true;
      } catch (e) {
        log.error({ error: serializeError(e) } as JsonObject, 'Failed to save snapshot');
        return false;
      }
    },
    [store, setClientState]
  );

  const restoreSnapshot = useCallback(
    async (key: string): Promise<RestoreResult> => {
      const req = dispatch(clientStateApi.endpoints.getClientStateByKey.initiate(key, { subscribe: false }));
      try {
        const raw = await req.unwrap();
        if (raw === null) {
          throw new Error('Snapshot data not found');
        }
        const parsed = JSON.parse(raw);
        const canvasState = zCanvasState.parse(parsed);

        // Check for missing images before restoring
        // Reuse the shared collectImageNames from canvasProjectFile — snapshots only
        // contain canvas entities (no global ref images), so we pass an empty array.
        const imageNames = collectImageNames(
          {
            rasterLayers: canvasState.rasterLayers.entities,
            controlLayers: canvasState.controlLayers.entities,
            inpaintMasks: canvasState.inpaintMasks.entities,
            regionalGuidance: canvasState.regionalGuidance.entities,
            bbox: canvasState.bbox,
            selectedEntityIdentifier: canvasState.selectedEntityIdentifier,
            bookmarkedEntityIdentifier: canvasState.bookmarkedEntityIdentifier,
          },
          []
        );
        const imageNamesList = [...imageNames];
        const missingImages = imageNamesList.length > 0 ? await findMissingImages(imageNamesList, dispatch) : [];

        if (missingImages.length > 0) {
          log.warn(
            { missingCount: missingImages.length, total: imageNamesList.length } as unknown as JsonObject,
            'Snapshot references images that no longer exist'
          );
        }

        dispatch(canvasSnapshotRestored(canvasState));
        return { success: true, missingImageCount: missingImages.length };
      } catch (e) {
        log.error({ error: serializeError(e) } as JsonObject, 'Failed to restore snapshot');
        // Distinguish Zod validation errors (incompatible snapshot) from other failures
        const isZodError = e instanceof z.ZodError;
        const isNotFound = e instanceof Error && e.message === 'Snapshot data not found';
        return {
          success: false,
          error: isZodError ? 'incompatible' : isNotFound ? 'not_found' : 'unknown',
        };
      } finally {
        req.unsubscribe();
      }
    },
    [dispatch]
  );

  const deleteSnapshot = useCallback(
    async (key: string) => {
      try {
        await deleteClientState(key).unwrap();
        return true;
      } catch (e) {
        log.error({ error: serializeError(e) } as JsonObject, 'Failed to delete snapshot');
        return false;
      }
    },
    [deleteClientState]
  );

  return {
    snapshots,
    saveSnapshot,
    restoreSnapshot,
    deleteSnapshot,
  };
};
