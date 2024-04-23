import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { setInfillMethod } from 'features/parameters/store/generationSlice';
import { appInfoApi } from 'services/api/endpoints/appInfo';

export const addAppConfigReceivedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: appInfoApi.endpoints.getAppConfig.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      const { infill_methods = [] } = action.payload;
      const infillMethod = getState().generation.infillMethod;

      if (!infill_methods.includes(infillMethod)) {
        // if there is no infill method, set it to the first one
        // if there is no first one... god help us
        dispatch(setInfillMethod(infill_methods[0] as string));
      }
    },
  });
};
