import { modelsApi } from 'services/api/endpoints/models';
import {
  socketModelInstallCompleted,
  socketModelInstallDownloading,
  socketModelInstallError,
} from 'services/events/actions';

import { startAppListening } from '../..';
import { api } from '../../../../../../services/api';

export const addModelInstallEventListener = () => {
  startAppListening({
    actionCreator: socketModelInstallDownloading,
    effect: async (action, { dispatch }) => {
      const { bytes, id } = action.payload.data;

      dispatch(
        modelsApi.util.updateQueryData('getModelImports', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.bytes = bytes;
            modelImport.status = 'downloading';
          }
          return draft;
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
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'completed';
          }
          return draft;
        })
      );
      dispatch(api.util.invalidateTags([{ type: "ModelConfig" }]))
    },
  });

  startAppListening({
    actionCreator: socketModelInstallError,
    effect: (action, { dispatch }) => {
      const { id, error_type } = action.payload.data;

      dispatch(
        modelsApi.util.updateQueryData('getModelImports', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'error';
            modelImport.error_reason = error_type
          }
          return draft;
        })
      );
    },
  });
};
