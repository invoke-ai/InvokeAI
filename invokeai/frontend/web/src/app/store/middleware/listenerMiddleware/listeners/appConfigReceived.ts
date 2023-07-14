import { setInfillMethod } from 'features/parameters/store/generationSlice';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import { startAppListening } from '..';

export const addAppConfigReceivedListener = () => {
  startAppListening({
    matcher: appInfoApi.endpoints.getAppConfig.matchFulfilled,
    effect: async (action, { getState, dispatch }) => {
      const { infill_methods } = action.payload;
      const infillMethod = getState().generation.infillMethod;

      if (!infill_methods.includes(infillMethod)) {
        dispatch(setInfillMethod(infill_methods[0]));
      }
    },
  });
};
