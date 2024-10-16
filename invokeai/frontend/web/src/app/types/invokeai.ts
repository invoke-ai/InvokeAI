import type { FilterType } from 'features/controlLayers/store/filters';
import type { ParameterPrecision, ParameterScheduler } from 'features/parameters/types/parameterSchemas';
import type { TabName } from 'features/ui/store/uiTypes';
import type { O } from 'ts-toolbelt';

/**
 * A disable-able application feature
 */
export type AppFeature =
  | 'faceRestore'
  | 'upscaling'
  | 'lightbox'
  | 'modelManager'
  | 'githubLink'
  | 'discordLink'
  | 'bugLink'
  | 'localization'
  | 'consoleLogging'
  | 'dynamicPrompting'
  | 'batches'
  | 'syncModels'
  | 'multiselect'
  | 'pauseQueue'
  | 'resumeQueue'
  | 'invocationCache'
  | 'bulkDownload'
  | 'starterModels'
  | 'hfToken';

/**
 * A disable-able Stable Diffusion feature
 */
export type SDFeature =
  | 'controlNet'
  | 'noise'
  | 'perlinNoise'
  | 'noiseThreshold'
  | 'variation'
  | 'symmetry'
  | 'seamless'
  | 'hires'
  | 'lora'
  | 'embedding'
  | 'vae'
  | 'hrf';

export type NumericalParameterConfig = {
  initial: number;
  sliderMin: number;
  sliderMax: number;
  numberInputMin: number;
  numberInputMax: number;
  fineStep: number;
  coarseStep: number;
};

/**
 * Configuration options for the InvokeAI UI.
 * Distinct from system settings which may be changed inside the app.
 */
export type AppConfig = {
  /**
   * Whether or not we should update image urls when image loading errors
   */
  shouldUpdateImagesOnConnect: boolean;
  shouldFetchMetadataFromApi: boolean;
  /**
   * Sets a size limit for outputs on the upscaling tab. This is a maximum dimension, so the actual max number of pixels
   * will be the square of this value.
   */
  maxUpscaleDimension?: number;
  allowPrivateBoards: boolean;
  allowPrivateStylePresets: boolean;
  disabledTabs: TabName[];
  disabledFeatures: AppFeature[];
  disabledSDFeatures: SDFeature[];
  nodesAllowlist: string[] | undefined;
  nodesDenylist: string[] | undefined;
  metadataFetchDebounce?: number;
  workflowFetchDebounce?: number;
  isLocal?: boolean;
  maxImageUploadCount?: number;
  sd: {
    defaultModel?: string;
    disabledControlNetModels: string[];
    disabledControlNetProcessors: FilterType[];
    // Core parameters
    iterations: NumericalParameterConfig;
    width: NumericalParameterConfig; // initial value comes from model
    height: NumericalParameterConfig; // initial value comes from model
    steps: NumericalParameterConfig;
    guidance: NumericalParameterConfig;
    cfgRescaleMultiplier: NumericalParameterConfig;
    img2imgStrength: NumericalParameterConfig;
    scheduler?: ParameterScheduler;
    vaePrecision?: ParameterPrecision;
    // Canvas
    boundingBoxHeight: NumericalParameterConfig; // initial value comes from model
    boundingBoxWidth: NumericalParameterConfig; // initial value comes from model
    scaledBoundingBoxHeight: NumericalParameterConfig; // initial value comes from model
    scaledBoundingBoxWidth: NumericalParameterConfig; // initial value comes from model
    canvasCoherenceStrength: NumericalParameterConfig;
    canvasCoherenceEdgeSize: NumericalParameterConfig;
    infillTileSize: NumericalParameterConfig;
    infillPatchmatchDownscaleSize: NumericalParameterConfig;
    // Misc advanced
    clipSkip: NumericalParameterConfig; // slider and input max are ignored for this, because the values depend on the model
    maskBlur: NumericalParameterConfig;
    hrfStrength: NumericalParameterConfig;
    dynamicPrompts: {
      maxPrompts: NumericalParameterConfig;
    };
    ca: {
      weight: NumericalParameterConfig;
    };
  };
  flux: {
    guidance: NumericalParameterConfig;
  };
};

export type PartialAppConfig = O.Partial<AppConfig, 'deep'>;
