import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import type { PartialAppConfig } from 'app/types/invokeai';
import { getDefaultAppConfig, zAppConfig } from 'app/types/invokeai';
import { merge } from 'es-toolkit/compat';
import z from 'zod';

const zConfigState = z.object({
  ...zAppConfig.shape,
  didLoad: z.boolean(),
});
type ConfigState = z.infer<typeof zConfigState>;

const getInitialState = (): ConfigState => ({
  ...getDefaultAppConfig(),
  didLoad: false,
});

const slice = createSlice({
  name: 'config',
  initialState: getInitialState(),
  reducers: {
    configChanged: (state, action: PayloadAction<PartialAppConfig>) => {
      // Handle disabledTabs specially - if provided, it should completely replace the default array
      if (action.payload.disabledTabs !== undefined) {
        state.disabledTabs = action.payload.disabledTabs;
      }

      // Merge the rest of the config normally
      merge(state, action.payload);
      state.didLoad = true;
    },
  },
});

export const { configChanged } = slice.actions;

export const configSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zConfigState,
  getInitialState,
};

export const selectConfigSlice = (state: RootState) => state.config;
const createConfigSelector = <T>(selector: Selector<ConfigState, T>) => createSelector(selectConfigSlice, selector);

export const selectWidthConfig = createConfigSelector((config) => config.sd.width);
export const selectHeightConfig = createConfigSelector((config) => config.sd.height);
export const selectStepsConfig = createConfigSelector((config) => config.sd.steps);
export const selectCFGScaleConfig = createConfigSelector((config) => config.sd.guidance);
export const selectGuidanceConfig = createConfigSelector((config) => config.flux.guidance);
export const selectCLIPSkipConfig = createConfigSelector((config) => config.sd.clipSkip);
export const selectCFGRescaleMultiplierConfig = createConfigSelector((config) => config.sd.cfgRescaleMultiplier);
export const selectCanvasCoherenceEdgeSizeConfig = createConfigSelector((config) => config.sd.canvasCoherenceEdgeSize);
export const selectMaskBlurConfig = createConfigSelector((config) => config.sd.maskBlur);
export const selectInfillPatchmatchDownscaleSizeConfig = createConfigSelector(
  (config) => config.sd.infillPatchmatchDownscaleSize
);
export const selectInfillTileSizeConfig = createConfigSelector((config) => config.sd.infillTileSize);
export const selectImg2imgStrengthConfig = createConfigSelector((config) => config.sd.img2imgStrength);
export const selectMaxPromptsConfig = createConfigSelector((config) => config.sd.dynamicPrompts.maxPrompts);
export const selectIterationsConfig = createConfigSelector((config) => config.sd.iterations);

export const selectMaxUpscaleDimension = createConfigSelector((config) => config.maxUpscaleDimension);
export const selectAllowPrivateStylePresets = createConfigSelector((config) => config.allowPrivateStylePresets);
export const selectWorkflowFetchDebounce = createConfigSelector((config) => config.workflowFetchDebounce ?? 300);
export const selectMetadataFetchDebounce = createConfigSelector((config) => config.metadataFetchDebounce ?? 300);

export const selectIsModelsTabDisabled = createConfigSelector((config) => config.disabledTabs.includes('models'));
export const selectIsClientSideUploadEnabled = createConfigSelector((config) => config.allowClientSideUpload);
export const selectAllowPublishWorkflows = createConfigSelector((config) => config.allowPublishWorkflows);
export const selectAllowPromptExpansion = createConfigSelector((config) => config.allowPromptExpansion);
export const selectAllowVideo = createConfigSelector((config) => config.allowVideo);

export const selectIsLocal = createSelector(selectConfigSlice, (config) => config.isLocal);
export const selectShouldShowCredits = createConfigSelector((config) => config.shouldShowCredits);
const selectDisabledTabs = createConfigSelector((config) => config.disabledTabs);
const selectDidLoad = createConfigSelector((config) => config.didLoad);
export const selectWithGenerateTab = createSelector(selectDidLoad, selectDisabledTabs, (didLoad, disabledTabs) =>
  didLoad ? !disabledTabs.includes('generate') : false
);
export const selectWithCanvasTab = createSelector(selectDidLoad, selectDisabledTabs, (didLoad, disabledTabs) =>
  didLoad ? !disabledTabs.includes('canvas') : false
);
export const selectWithUpscalingTab = createSelector(selectDidLoad, selectDisabledTabs, (didLoad, disabledTabs) =>
  didLoad ? !disabledTabs.includes('upscaling') : false
);
export const selectWithWorkflowsTab = createSelector(selectDidLoad, selectDisabledTabs, (didLoad, disabledTabs) =>
  didLoad ? !disabledTabs.includes('workflows') : false
);
export const selectWithModelsTab = createSelector(selectDidLoad, selectDisabledTabs, (didLoad, disabledTabs) =>
  didLoad ? !disabledTabs.includes('models') : false
);
export const selectWithQueueTab = createSelector(selectDidLoad, selectDisabledTabs, (didLoad, disabledTabs) =>
  didLoad ? !disabledTabs.includes('queue') : false
);
export const selectWithVideoTab = createSelector(selectDidLoad, selectDisabledTabs, (didLoad, disabledTabs) =>
  didLoad ? !disabledTabs.includes('video') : false
);
