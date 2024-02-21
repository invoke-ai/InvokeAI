import { logger } from 'app/logging/logger';
import {
  socketModelInstallCompleted,
  socketModelInstallDownloading,
  socketModelInstallError,
} from 'services/events/actions';
import { modelsApi } from 'services/api/endpoints/models';
import type { components, paths } from 'services/api/schema';


import { startAppListening } from '../..';
import { createEntityAdapter } from '@reduxjs/toolkit';

const log = logger('socketio');

export const addModelInstallEventListener = () => {
  startAppListening({
    actionCreator: socketModelInstallDownloading,
    effect: async (action, { dispatch }) => {
      const { bytes, local_path, source, timestamp, total_bytes } = action.payload.data;
      let message = `Model install started: ${bytes}/${total_bytes}/${source}`;
    // below doesnt work, still not sure how to update the importModels data 
    //   dispatch(
    //       modelsApi.util.updateQueryData('getModelImports', undefined, (draft) => {
    //         importModelsAdapter.updateOne(draft, {
    //             id: source,
    //             changes: {
    //                 bytes,
    //                 total_bytes,
    //             },\q
    //             });
    //         }
    //         )
    //   );

      log.debug(action.payload, message);
    },
  });

  startAppListening({
    actionCreator: socketModelInstallCompleted,
    effect: (action) => {
      const { key, source, timestamp } = action.payload.data;

      let message = `Model install completed: ${source}`;

    //   dispatch something that marks the model installation as completed

      log.debug(action.payload, message);
    },
  });

  startAppListening({
    actionCreator: socketModelInstallError,
    effect: (action) => {
      const { error, error_type, source } = action.payload.data;

    //   dispatch something that marks the model installation as errored

      log.debug(action.payload, error);
    },
  });
};
