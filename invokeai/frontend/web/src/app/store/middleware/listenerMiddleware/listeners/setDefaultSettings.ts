import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { setDefaultSettings } from 'features/parameters/store/actions';
import {
  setCfgRescaleMultiplier,
  setCfgScale,
  setScheduler,
  setSteps,
  vaePrecisionChanged,
  vaeSelected,
} from 'features/parameters/store/generationSlice';
import {
  isParameterCFGRescaleMultiplier,
  isParameterCFGScale,
  isParameterPrecision,
  isParameterScheduler,
  isParameterSteps,
  zParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { map } from 'lodash-es';
import { modelsApi } from 'services/api/endpoints/models';
import { isNonRefinerMainModelConfig } from 'services/api/types';

export const addSetDefaultSettingsListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: setDefaultSettings,
    effect: async (action, { dispatch, getState }) => {
      const state = getState();

      const currentModel = state.generation.model;

      if (!currentModel) {
        return;
      }

      const modelConfig = await dispatch(modelsApi.endpoints.getModelConfig.initiate(currentModel.key)).unwrap();

      if (!modelConfig) {
        return;
      }

      if (isNonRefinerMainModelConfig(modelConfig) && modelConfig.default_settings) {
        const { vae, vae_precision, cfg_scale, cfg_rescale_multiplier, steps, scheduler } =
          modelConfig.default_settings;

        if (vae) {
          // we store this as "default" within default settings
          // to distinguish it from no default set
          if (vae === 'default') {
            dispatch(vaeSelected(null));
          } else {
            const { data } = modelsApi.endpoints.getVaeModels.select()(state);
            const vaeArray = map(data?.entities);
            const validVae = vaeArray.find((model) => model.key === vae);

            const result = zParameterVAEModel.safeParse(validVae);
            if (!result.success) {
              return;
            }
            dispatch(vaeSelected(result.data));
          }
        }

        if (vae_precision) {
          if (isParameterPrecision(vae_precision)) {
            dispatch(vaePrecisionChanged(vae_precision));
          }
        }

        if (cfg_scale) {
          if (isParameterCFGScale(cfg_scale)) {
            dispatch(setCfgScale(cfg_scale));
          }
        }

        if (cfg_rescale_multiplier) {
          if (isParameterCFGRescaleMultiplier(cfg_rescale_multiplier)) {
            dispatch(setCfgRescaleMultiplier(cfg_rescale_multiplier));
          }
        }

        if (steps) {
          if (isParameterSteps(steps)) {
            dispatch(setSteps(steps));
          }
        }

        if (scheduler) {
          if (isParameterScheduler(scheduler)) {
            dispatch(setScheduler(scheduler));
          }
        }

        dispatch(addToast(makeToast({ title: t('toast.parameterSet', { parameter: 'Default settings' }) })));
      }
    },
  });
};
