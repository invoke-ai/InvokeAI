import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { buildAdHocUpscaleGraph } from 'features/nodes/util/graphBuilders/buildAdHocUpscaleGraph';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { startAppListening } from '..';
import { ImageDTO } from 'services/api/types';
import { createIsAllowedToUpscaleSelector } from 'features/parameters/hooks/useIsAllowedToUpscale';

export const upscaleRequested = createAction<{ imageDTO: ImageDTO }>(
  `upscale/upscaleRequested`
);

export const addUpscaleRequestedListener = () => {
  startAppListening({
    actionCreator: upscaleRequested,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('session');

      const { imageDTO } = action.payload;
      const { image_name } = imageDTO;
      const state = getState();

      const { isAllowedToUpscale, detailTKey } =
        createIsAllowedToUpscaleSelector(imageDTO)(state);

      // if we can't upscale, show a toast and return
      if (!isAllowedToUpscale) {
        log.error(
          { imageDTO },
          t(detailTKey ?? 'parameters.isAllowedToUpscale.tooLarge') // should never coalesce
        );
        dispatch(
          addToast({
            title: t(detailTKey ?? 'parameters.isAllowedToUpscale.tooLarge'), // should never coalesce
            status: 'error',
          })
        );
        return;
      }

      const { esrganModelName } = state.postprocessing;
      const { autoAddBoardId } = state.gallery;

      const graph = buildAdHocUpscaleGraph({
        image_name,
        esrganModelName,
        autoAddBoardId,
      });

      try {
        const req = dispatch(
          queueApi.endpoints.enqueueGraph.initiate(
            { graph, prepend: true },
            {
              fixedCacheKey: 'enqueueGraph',
            }
          )
        );

        const enqueueResult = await req.unwrap();
        req.reset();
        log.debug(
          { enqueueResult: parseify(enqueueResult) },
          t('queue.graphQueued')
        );
      } catch (error) {
        log.error({ graph: parseify(graph) }, t('queue.graphFailedToQueue'));

        // handle usage-related errors
        if (error instanceof Object) {
          if ('data' in error && 'status' in error) {
            if (error.status === 403) {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              const detail = (error.data as any)?.detail || 'Unknown Error';
              dispatch(
                addToast({
                  title: t('queue.graphFailedToQueue'),
                  status: 'error',
                  description: detail,
                  duration: 15000,
                })
              );
              return;
            }
          }
        }

        dispatch(
          addToast({
            title: t('queue.graphFailedToQueue'),
            status: 'error',
          })
        );
      }
    },
  });
};
