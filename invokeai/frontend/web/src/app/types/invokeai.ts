import type { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { InvokeTabName } from 'features/ui/store/tabMap';
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
  | 'prependQueue'
  | 'invocationCache'
  | 'bulkDownload';
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
  disabledTabs: InvokeTabName[];
  disabledFeatures: AppFeature[];
  disabledSDFeatures: SDFeature[];
  canRestoreDeletedImagesFromBin: boolean;
  nodesAllowlist: string[] | undefined;
  nodesDenylist: string[] | undefined;
  maxUpscalePixels?: number;
  metadataFetchDebounce?: number;
  workflowFetchDebounce?: number;
  sd: {
    defaultModel?: string;
    disabledControlNetModels: string[];
    disabledControlNetProcessors: (keyof typeof CONTROLNET_PROCESSORS)[];
    // Core parameters
    iterations: NumericalParameterConfig;
    width: NumericalParameterConfig; // initial value comes from model
    height: NumericalParameterConfig; // initial value comes from model
    steps: NumericalParameterConfig;
    guidance: NumericalParameterConfig;
    cfgRescaleMultiplier: NumericalParameterConfig;
    img2imgStrength: NumericalParameterConfig;
    // Canvas
    boundingBoxHeight: NumericalParameterConfig; // initial value comes from model
    boundingBoxWidth: NumericalParameterConfig; // initial value comes from model
    scaledBoundingBoxHeight: NumericalParameterConfig; // initial value comes from model
    scaledBoundingBoxWidth: NumericalParameterConfig; // initial value comes from model
    canvasCoherenceStrength: NumericalParameterConfig;
    canvasCoherenceSteps: NumericalParameterConfig;
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
};

export type PartialAppConfig = O.Partial<AppConfig, 'deep'>;
