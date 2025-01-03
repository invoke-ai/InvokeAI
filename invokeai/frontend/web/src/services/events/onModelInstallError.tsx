import { logger } from 'app/logging/logger';
import type { AppDispatch, RootState } from 'app/store/store';
import { $isHFForbiddenToastOpen } from 'features/modelManagerV2/hooks/useHFForbiddenToast';
import { $isHFLoginToastOpen } from 'features/modelManagerV2/hooks/useHFLoginToast';
import { api } from 'services/api';
import { modelsApi } from 'services/api/endpoints/models';
import type { S } from 'services/api/types';

const log = logger('events');
const selectModelInstalls = modelsApi.endpoints.listModelInstalls.select();

export const buildOnModelInstallError = (getState: () => RootState, dispatch: AppDispatch) => {
  return (data: S['ModelInstallErrorEvent']) => {
    log.error({ data }, 'Model install error');

    const { id, error, error_type } = data;
    const installs = selectModelInstalls(getState()).data;

    if (error === 'Unauthorized') {
      $isHFLoginToastOpen.set(true);
    }

    if (error === 'Forbidden') {
      $isHFForbiddenToastOpen.set({ isEnabled: true, source: data.source });
    }

    if (!installs?.find((install) => install.id === id)) {
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
  };
};
