import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { O } from 'ts-toolbelt';

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
  sd: {
    defaultModel?: string;
    disabledControlNetModels: string[];
    disabledControlNetProcessors: (keyof typeof CONTROLNET_PROCESSORS)[];
    iterations: {
      initial: number;
      min: number;
      sliderMax: number;
      inputMax: number;
      fineStep: number;
      coarseStep: number;
    };
    width: {
      initial: number;
      min: number;
      sliderMax: number;
      inputMax: number;
      fineStep: number;
      coarseStep: number;
    };
    height: {
      initial: number;
      min: number;
      sliderMax: number;
      inputMax: number;
      fineStep: number;
      coarseStep: number;
    };
    steps: {
      initial: number;
      min: number;
      sliderMax: number;
      inputMax: number;
      fineStep: number;
      coarseStep: number;
    };
    guidance: {
      initial: number;
      min: number;
      sliderMax: number;
      inputMax: number;
      fineStep: number;
      coarseStep: number;
    };
    img2imgStrength: {
      initial: number;
      min: number;
      sliderMax: number;
      inputMax: number;
      fineStep: number;
      coarseStep: number;
    };
    hrfStrength: {
      initial: number;
      min: number;
      sliderMax: number;
      inputMax: number;
      fineStep: number;
      coarseStep: number;
    };
    dynamicPrompts: {
      maxPrompts: {
        initial: number;
        min: number;
        sliderMax: number;
        inputMax: number;
      };
    };
  };
};

export type PartialAppConfig = O.Partial<AppConfig, 'deep'>;
