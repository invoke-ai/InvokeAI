import { InvokeTabName } from 'features/ui/store/tabMap';
import { O } from 'ts-toolbelt';

// These are old types from the model management UI

// export type ModelStatus = 'active' | 'cached' | 'not loaded';

// export type Model = {
//   status: ModelStatus;
//   description: string;
//   weights: string;
//   config?: string;
//   vae?: string;
//   width?: number;
//   height?: number;
//   default?: boolean;
//   format?: string;
// };

// export type DiffusersModel = {
//   status: ModelStatus;
//   description: string;
//   repo_id?: string;
//   path?: string;
//   vae?: {
//     repo_id?: string;
//     path?: string;
//   };
//   format?: string;
//   default?: boolean;
// };

// export type ModelList = Record<string, Model & DiffusersModel>;

// export type FoundModel = {
//   name: string;
//   location: string;
// };

// export type InvokeModelConfigProps = {
//   name: string | undefined;
//   description: string | undefined;
//   config: string | undefined;
//   weights: string | undefined;
//   vae: string | undefined;
//   width: number | undefined;
//   height: number | undefined;
//   default: boolean | undefined;
//   format: string | undefined;
// };

// export type InvokeDiffusersModelConfigProps = {
//   name: string | undefined;
//   description: string | undefined;
//   repo_id: string | undefined;
//   path: string | undefined;
//   default: boolean | undefined;
//   format: string | undefined;
//   vae: {
//     repo_id: string | undefined;
//     path: string | undefined;
//   };
// };

// export type InvokeModelConversionProps = {
//   model_name: string;
//   save_location: string;
//   custom_location: string | null;
// };

// export type InvokeModelMergingProps = {
//   models_to_merge: string[];
//   alpha: number;
//   interp: 'weighted_sum' | 'sigmoid' | 'inv_sigmoid' | 'add_difference';
//   force: boolean;
//   merged_model_name: string;
//   model_merge_save_path: string | null;
// };

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
  | 'consoleLogging';

/**
 * A disable-able Stable Diffusion feature
 */
export type SDFeature =
  | 'noise'
  | 'variation'
  | 'symmetry'
  | 'seamless'
  | 'hires';

/**
 * Configuration options for the InvokeAI UI.
 * Distinct from system settings which may be changed inside the app.
 */
export type AppConfig = {
  /**
   * Whether or not URLs should be transformed to use a different host
   */
  shouldTransformUrls: boolean;
  /**
   * Whether or not we need to re-fetch images
   */
  disabledTabs: InvokeTabName[];
  disabledFeatures: AppFeature[];
  disabledSDFeatures: SDFeature[];
  canRestoreDeletedImagesFromBin: boolean;
  sd: {
    defaultModel?: string;
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
  };
};

export type PartialAppConfig = O.Partial<AppConfig, 'deep'>;
