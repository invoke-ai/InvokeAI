import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { setInfillMethod } from 'features/controlLayers/store/paramsSlice';
import { shouldUseNSFWCheckerChanged, shouldUseWatermarkerChanged } from 'features/system/store/systemSlice';
import { appInfoApi } from 'services/api/endpoints/appInfo';

export const addAppConfigReceivedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: appInfoApi.endpoints.getAppConfig.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      const { infill_methods = [], nsfw_methods = [], watermarking_methods = [] } = action.payload;
      const infillMethod = getState().params.infillMethod;

      if (!infill_methods.includes(infillMethod)) {
        // If the selected infill method does not exist, prefer 'lama' if it's in the list, otherwise 'tile'.
        // TODO(psyche): lama _should_ always be in the list, but the API doesn't guarantee it...
        const infillMethod = infill_methods.includes('lama') ? 'lama' : 'tile';
        dispatch(setInfillMethod(infillMethod));
      }

      if (!nsfw_methods.includes('nsfw_checker')) {
        dispatch(shouldUseNSFWCheckerChanged(false));
      }

      if (!watermarking_methods.includes('invisible_watermark')) {
        dispatch(shouldUseWatermarkerChanged(false));
      }
    },
  });
};
