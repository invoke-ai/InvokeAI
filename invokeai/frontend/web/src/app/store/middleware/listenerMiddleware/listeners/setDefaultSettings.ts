import type { AppStartListening } from 'app/store/store';
import { isNil } from 'es-toolkit';
import { bboxHeightChanged, bboxWidthChanged } from 'features/controlLayers/store/canvasSlice';
import {
  buildSelectIsStagingBySessionId,
  selectActiveCanvasSessionId,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import {
  heightChanged,
  paramsDispatch,
  selectActiveParams,
  setCfgRescaleMultiplier,
  setCfgScale,
  setGuidance,
  setScheduler,
  setSteps,
  vaePrecisionChanged,
  vaeSelected,
  widthChanged,
} from 'features/controlLayers/store/paramsSlice';
import { setDefaultSettings } from 'features/parameters/store/actions';
import {
  isParameterCFGRescaleMultiplier,
  isParameterCFGScale,
  isParameterGuidance,
  isParameterHeight,
  isParameterPrecision,
  isParameterScheduler,
  isParameterSteps,
  isParameterWidth,
  zParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { t } from 'i18next';
import { modelConfigsAdapterSelectors, modelsApi } from 'services/api/endpoints/models';
import { isNonRefinerMainModelConfig } from 'services/api/types';

export const addSetDefaultSettingsListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: setDefaultSettings,
    effect: async (action, api) => {
      const { dispatch, getState } = api;
      const state = getState();

      const currentModel = selectActiveParams(state).model;

      if (!currentModel) {
        return;
      }

      const request = dispatch(modelsApi.endpoints.getModelConfigs.initiate(undefined, { subscribe: false }));
      const data = await request.unwrap();
      const models = modelConfigsAdapterSelectors.selectAll(data);

      const modelConfig = models.find((model) => model.key === currentModel.key);

      if (!modelConfig) {
        return;
      }

      if (isNonRefinerMainModelConfig(modelConfig) && modelConfig.default_settings) {
        const { vae, vae_precision, cfg_scale, cfg_rescale_multiplier, steps, scheduler, width, height, guidance } =
          modelConfig.default_settings;

        if (vae) {
          // we store this as "default" within default settings
          // to distinguish it from no default set
          if (vae === 'default') {
            paramsDispatch(api, vaeSelected, null);
          } else {
            const vaeModel = models.find((model) => model.key === vae);
            const result = zParameterVAEModel.safeParse(vaeModel);
            if (!result.success) {
              return;
            }
            paramsDispatch(api, vaeSelected, result.data);
          }
        }

        if (vae_precision) {
          if (isParameterPrecision(vae_precision)) {
            paramsDispatch(api, vaePrecisionChanged, vae_precision);
          }
        }

        if (guidance) {
          if (isParameterGuidance(guidance)) {
            paramsDispatch(api, setGuidance, guidance);
          }
        }

        if (cfg_scale) {
          if (isParameterCFGScale(cfg_scale)) {
            paramsDispatch(api, setCfgScale, cfg_scale);
          }
        }

        if (!isNil(cfg_rescale_multiplier)) {
          if (isParameterCFGRescaleMultiplier(cfg_rescale_multiplier)) {
            paramsDispatch(api, setCfgRescaleMultiplier, cfg_rescale_multiplier);
          }
        } else {
          // Set this to 0 if it doesn't have a default. This value is
          // easy to miss in the UI when users are resetting defaults
          // and leaving it non-zero could lead to detrimental
          // effects.
          paramsDispatch(api, setCfgRescaleMultiplier, 0);
        }

        if (steps) {
          if (isParameterSteps(steps)) {
            paramsDispatch(api, setSteps, steps);
          }
        }

        if (scheduler) {
          if (isParameterScheduler(scheduler)) {
            paramsDispatch(api, setScheduler, scheduler);
          }
        }
        const setSizeOptions = { updateAspectRatio: true, clamp: true };

        const sessionId = selectActiveCanvasSessionId(state);
        const selectIsStaging = buildSelectIsStagingBySessionId(sessionId);
        const isStaging = selectIsStaging(state);

        const activeTab = selectActiveTab(getState());
        if (activeTab === 'generate') {
          if (isParameterWidth(width)) {
            paramsDispatch(api, widthChanged, { width, ...setSizeOptions });
          }
          if (isParameterHeight(height)) {
            paramsDispatch(api, heightChanged, { height, ...setSizeOptions });
          }
        }

        if (activeTab === 'canvas') {
          if (!isStaging) {
            if (isParameterWidth(width)) {
              dispatch(bboxWidthChanged({ width, ...setSizeOptions }));
            }
            if (isParameterHeight(height)) {
              dispatch(bboxHeightChanged({ height, ...setSizeOptions }));
            }
          }
        }

        toast({ id: 'PARAMETER_SET', title: t('toast.parameterSet', { parameter: 'Default settings' }) });
      }
    },
  });
};
