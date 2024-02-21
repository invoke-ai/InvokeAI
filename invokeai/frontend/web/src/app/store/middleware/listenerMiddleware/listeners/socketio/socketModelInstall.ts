import { modelsApi } from 'services/api/endpoints/models';
import {
  socketModelInstallCompleted,
  socketModelInstallDownloading,
  socketModelInstallError,
} from 'services/events/actions';

import { startAppListening } from '../..';

export const addModelInstallEventListener = () => {
  startAppListening({
    actionCreator: socketModelInstallDownloading,
    effect: async (action, { dispatch }) => {
      const { bytes, id } = action.payload.data;

      dispatch(
        modelsApi.util.updateQueryData('getModelImports', undefined, (draft) => {
          const models = JSON.parse(JSON.stringify(draft))

          const modelIndex = models.findIndex((m) => m.id === id);

          models[modelIndex].bytes = bytes;
          models[modelIndex].status = 'downloading';
          return models;
        })
      );
    },
  });

  startAppListening({
    actionCreator: socketModelInstallCompleted,
    effect: (action, { dispatch }) => {
      const { id } = action.payload.data;

      dispatch(
        modelsApi.util.updateQueryData('getModelImports', undefined, (draft) => {
          const models = JSON.parse(JSON.stringify(draft))

          const modelIndex = models.findIndex((m) => m.id === id);

          models[modelIndex].status = 'completed';
          return models;
        })
      );
    },
  });

  startAppListening({
    actionCreator: socketModelInstallError,
    effect: (action, { dispatch }) => {
      const { id } = action.payload.data;

      dispatch(
        modelsApi.util.updateQueryData('getModelImports', undefined, (draft) => {
          const models = JSON.parse(JSON.stringify(draft))

          const modelIndex = models.findIndex((m) => m.id === id);

          models[modelIndex].status = 'error';
          return models;
        })
      );
    },
  });
};
