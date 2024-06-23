import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { api, LIST_TAG } from 'services/api';
import { modelsApi } from 'services/api/endpoints/models';
import {
  socketModelInstallCancelled,
  socketModelInstallComplete,
  socketModelInstallDownloadProgress,
  socketModelInstallDownloadsComplete,
  socketModelInstallDownloadStarted,
  socketModelInstallError,
  socketModelInstallStarted,
} from 'services/events/actions';

/**
 * A model install has two main stages - downloading and installing. All these events are namespaced under `model_install_`
 * which is a bit misleading. For example, a `model_install_started` event is actually fired _after_ the model has fully
 * downloaded and is being "physically" installed.
 *
 * Note: the download events are only fired for remote model installs, not local.
 *
 * Here's the expected flow:
 * - API receives install request, model manager preps the install
 * - `model_install_download_started` fired when the download starts
 * - `model_install_download_progress` fired continually until the download is complete
 * - `model_install_download_complete` fired when the download is complete
 * - `model_install_started` fired when the "physical" installation starts
 * - `model_install_complete` fired when the installation is complete
 * - `model_install_cancelled` fired if the installation is cancelled
 * - `model_install_error` fired if the installation has an error
 */

const selectModelInstalls = modelsApi.endpoints.listModelInstalls.select();

export const addModelInstallEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketModelInstallDownloadStarted,
    effect: async (action, { dispatch, getState }) => {
      const { id } = action.payload.data;
      const { data } = selectModelInstalls(getState());

      if (!data || !data.find((m) => m.id === id)) {
        dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
      } else {
        dispatch(
          modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
            const modelImport = draft.find((m) => m.id === id);
            if (modelImport) {
              modelImport.status = 'downloading';
            }
            return draft;
          })
        );
      }
    },
  });

  startAppListening({
    actionCreator: socketModelInstallStarted,
    effect: async (action, { dispatch, getState }) => {
      const { id } = action.payload.data;
      const { data } = selectModelInstalls(getState());

      if (!data || !data.find((m) => m.id === id)) {
        dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
      } else {
        dispatch(
          modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
            const modelImport = draft.find((m) => m.id === id);
            if (modelImport) {
              modelImport.status = 'running';
            }
            return draft;
          })
        );
      }
    },
  });

  startAppListening({
    actionCreator: socketModelInstallDownloadProgress,
    effect: async (action, { dispatch, getState }) => {
      const { bytes, total_bytes, id } = action.payload.data;
      const { data } = selectModelInstalls(getState());

      if (!data || !data.find((m) => m.id === id)) {
        dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
      } else {
        dispatch(
          modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
            const modelImport = draft.find((m) => m.id === id);
            if (modelImport) {
              modelImport.bytes = bytes;
              modelImport.total_bytes = total_bytes;
              modelImport.status = 'downloading';
            }
            return draft;
          })
        );
      }
    },
  });

  startAppListening({
    actionCreator: socketModelInstallComplete,
    effect: (action, { dispatch, getState }) => {
      const { id } = action.payload.data;

      const { data } = selectModelInstalls(getState());

      if (!data || !data.find((m) => m.id === id)) {
        dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
      } else {
        dispatch(
          modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
            const modelImport = draft.find((m) => m.id === id);
            if (modelImport) {
              modelImport.status = 'completed';
            }
            return draft;
          })
        );
      }

      dispatch(api.util.invalidateTags([{ type: 'ModelConfig', id: LIST_TAG }]));
      dispatch(api.util.invalidateTags([{ type: 'ModelScanFolderResults', id: LIST_TAG }]));
    },
  });

  startAppListening({
    actionCreator: socketModelInstallError,
    effect: (action, { dispatch, getState }) => {
      const { id, error, error_type } = action.payload.data;
      const { data } = selectModelInstalls(getState());

      if (!data || !data.find((m) => m.id === id)) {
        dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
      } else {
        dispatch(
          modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
            const modelImport = draft.find((m) => m.id === id);
            if (modelImport) {
              modelImport.status = 'error';
              modelImport.error_reason = error_type;
              modelImport.error = error;
            }
            return draft;
          })
        );
      }
    },
  });

  startAppListening({
    actionCreator: socketModelInstallCancelled,
    effect: (action, { dispatch, getState }) => {
      const { id } = action.payload.data;
      const { data } = selectModelInstalls(getState());

      if (!data || !data.find((m) => m.id === id)) {
        dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
      } else {
        dispatch(
          modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
            const modelImport = draft.find((m) => m.id === id);
            if (modelImport) {
              modelImport.status = 'cancelled';
            }
            return draft;
          })
        );
      }
    },
  });

  startAppListening({
    actionCreator: socketModelInstallDownloadsComplete,
    effect: (action, { dispatch, getState }) => {
      const { id } = action.payload.data;
      const { data } = selectModelInstalls(getState());

      if (!data || !data.find((m) => m.id === id)) {
        dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
      } else {
        dispatch(
          modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
            const modelImport = draft.find((m) => m.id === id);
            if (modelImport) {
              modelImport.status = 'downloads_done';
            }
            return draft;
          })
        );
      }
    },
  });
};
