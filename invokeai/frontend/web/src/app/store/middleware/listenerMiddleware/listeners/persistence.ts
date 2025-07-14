import { createAction } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { $isPendingPersist, persistConfigs } from 'app/store/store';
import { setInfillMethod } from 'features/controlLayers/store/paramsSlice';
import { shouldUseNSFWCheckerChanged, shouldUseWatermarkerChanged } from 'features/system/store/systemSlice';
import { REMEMBER_PERSISTED } from 'redux-remember';
import { appInfoApi } from 'services/api/endpoints/appInfo';

export const addPersistenceListener = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action, currentState, originalState) => {
      for (const persistConfig of Object.values(persistConfigs)) {
        const originalSlice = originalState[persistConfig.name];
        const currentSlice = currentState[persistConfig.name];
        const allKeys = Object.keys(currentSlice);
        const persistedKeys = allKeys.filter((k) => !persistConfig.persistDenylist.includes(k));
        for (const key of persistedKeys) {
          if (currentSlice[key as keyof typeof currentSlice] !== originalSlice[key as keyof typeof originalSlice]) {
            return true;
          }
        }
      }
      return false;
    },
    effect: () => {
      $isPendingPersist.set(true);
    },
  });

  startAppListening({
    matcher: createAction(REMEMBER_PERSISTED).match,
    effect: () => {
      $isPendingPersist.set(false);
    },
  });
};
