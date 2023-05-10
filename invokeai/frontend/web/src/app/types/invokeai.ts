/**
 * Types for images, the things they are made of, and the things
 * they make up.
 *
 * Generated images are txt2img and img2img images. They may have
 * had additional postprocessing done on them when they were first
 * generated.
 *
 * Postprocessed images are images which were not generated here
 * but only postprocessed by the app. They only get postprocessing
 * metadata and have a different image type, e.g. 'esrgan' or
 * 'gfpgan'.
 */

import { GalleryCategory } from 'features/gallery/store/gallerySlice';
import { FacetoolType } from 'features/parameters/store/postprocessingSlice';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { IRect } from 'konva/lib/types';
import { ImageResponseMetadata, ImageType } from 'services/api';
import { AnyInvocation } from 'services/events/types';
import { O } from 'ts-toolbelt';

/**
 * TODO:
 * Once an image has been generated, if it is postprocessed again,
 * additional postprocessing steps are added to its postprocessing
 * array.
 *
 * TODO: Better documentation of types.
 */

export type PromptItem = {
  prompt: string;
  weight: number;
};

// TECHDEBT: We need to retain compatibility with plain prompt strings and the structure Prompt type
export type Prompt = Array<PromptItem> | string;

export type SeedWeightPair = {
  seed: number;
  weight: number;
};

export type SeedWeights = Array<SeedWeightPair>;

// All generated images contain these metadata.
export type CommonGeneratedImageMetadata = {
  postprocessing: null | Array<ESRGANMetadata | FacetoolMetadata>;
  sampler:
    | 'ddim'
    | 'k_dpm_2_a'
    | 'k_dpm_2'
    | 'k_dpmpp_2_a'
    | 'k_dpmpp_2'
    | 'k_euler_a'
    | 'k_euler'
    | 'k_heun'
    | 'k_lms'
    | 'plms';
  prompt: Prompt;
  seed: number;
  variations: SeedWeights;
  steps: number;
  cfg_scale: number;
  width: number;
  height: number;
  seamless: boolean;
  hires_fix: boolean;
  extra: null | Record<string, never>; // Pending development of RFC #266
};

// txt2img and img2img images have some unique attributes.
export type Txt2ImgMetadata = CommonGeneratedImageMetadata & {
  type: 'txt2img';
};

export type Img2ImgMetadata = CommonGeneratedImageMetadata & {
  type: 'img2img';
  orig_hash: string;
  strength: number;
  fit: boolean;
  init_image_path: string;
  mask_image_path?: string;
};

// Superset of  generated image metadata types.
export type GeneratedImageMetadata = Txt2ImgMetadata | Img2ImgMetadata;

// All post processed images contain these metadata.
export type CommonPostProcessedImageMetadata = {
  orig_path: string;
  orig_hash: string;
};

// esrgan and gfpgan images have some unique attributes.
export type ESRGANMetadata = CommonPostProcessedImageMetadata & {
  type: 'esrgan';
  scale: 2 | 4;
  strength: number;
  denoise_str: number;
};

export type FacetoolMetadata = CommonPostProcessedImageMetadata & {
  type: 'gfpgan' | 'codeformer';
  strength: number;
  fidelity?: number;
};

// Superset of all postprocessed image metadata types..
export type PostProcessedImageMetadata = ESRGANMetadata | FacetoolMetadata;

// Metadata includes the system config and image metadata.
// export type Metadata = SystemGenerationMetadata & {
//   image: GeneratedImageMetadata | PostProcessedImageMetadata;
// };

/**
 * ResultImage
 */
export type Image = {
  name: string;
  type: ImageType;
  url: string;
  thumbnail: string;
  metadata: ImageResponseMetadata;
};

/**
 * Types related to the system status.
 */

// // This represents the processing status of the backend.
// export type SystemStatus = {
//   isProcessing: boolean;
//   currentStep: number;
//   totalSteps: number;
//   currentIteration: number;
//   totalIterations: number;
//   currentStatus: string;
//   currentStatusHasSteps: boolean;
//   hasError: boolean;
// };

// export type SystemGenerationMetadata = {
//   model: string;
//   model_weights?: string;
//   model_id?: string;
//   model_hash: string;
//   app_id: string;
//   app_version: string;
// };

// export type SystemConfig = SystemGenerationMetadata & {
//   model_list: ModelList;
//   infill_methods: string[];
// };

export type ModelStatus = 'active' | 'cached' | 'not loaded';

export type Model = {
  status: ModelStatus;
  description: string;
  weights: string;
  config?: string;
  vae?: string;
  width?: number;
  height?: number;
  default?: boolean;
  format?: string;
};

export type DiffusersModel = {
  status: ModelStatus;
  description: string;
  repo_id?: string;
  path?: string;
  vae?: {
    repo_id?: string;
    path?: string;
  };
  format?: string;
  default?: boolean;
};

export type ModelList = Record<string, Model & DiffusersModel>;

export type FoundModel = {
  name: string;
  location: string;
};

export type InvokeModelConfigProps = {
  name: string | undefined;
  description: string | undefined;
  config: string | undefined;
  weights: string | undefined;
  vae: string | undefined;
  width: number | undefined;
  height: number | undefined;
  default: boolean | undefined;
  format: string | undefined;
};

export type InvokeDiffusersModelConfigProps = {
  name: string | undefined;
  description: string | undefined;
  repo_id: string | undefined;
  path: string | undefined;
  default: boolean | undefined;
  format: string | undefined;
  vae: {
    repo_id: string | undefined;
    path: string | undefined;
  };
};

export type InvokeModelConversionProps = {
  model_name: string;
  save_location: string;
  custom_location: string | null;
};

export type InvokeModelMergingProps = {
  models_to_merge: string[];
  alpha: number;
  interp: 'weighted_sum' | 'sigmoid' | 'inv_sigmoid' | 'add_difference';
  force: boolean;
  merged_model_name: string;
  model_merge_save_path: string | null;
};

/**
 * These types type data received from the server via socketio.
 */

export type ModelChangeResponse = {
  model_name: string;
  model_list: ModelList;
};

export type ModelConvertedResponse = {
  converted_model_name: string;
  model_list: ModelList;
};

export type ModelsMergedResponse = {
  merged_models: string[];
  merged_model_name: string;
  model_list: ModelList;
};

export type ModelAddedResponse = {
  new_model_name: string;
  model_list: ModelList;
  update: boolean;
};

export type ModelDeletedResponse = {
  deleted_model_name: string;
  model_list: ModelList;
};

export type FoundModelResponse = {
  search_folder: string;
  found_models: FoundModel[];
};

// export type SystemStatusResponse = SystemStatus;

// export type SystemConfigResponse = SystemConfig;

export type ImageResultResponse = Omit<_Image, 'uuid'> & {
  boundingBox?: IRect;
  generationMode: InvokeTabName;
};

export type ImageUploadResponse = {
  // image: Omit<Image, 'uuid' | 'metadata' | 'category'>;
  url: string;
  mtime: number;
  width: number;
  height: number;
  thumbnail: string;
  // bbox: [number, number, number, number];
};

export type ErrorResponse = {
  message: string;
  additionalData?: string;
};

export type ImageUrlResponse = {
  url: string;
};

export type UploadOutpaintingMergeImagePayload = {
  dataURL: string;
  name: string;
};

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
  | 'localization';

/**
 * A disable-able Stable Diffusion feature
 */
export type StableDiffusionFeature =
  | 'noiseConfig'
  | 'variations'
  | 'symmetry'
  | 'tiling'
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
  shouldFetchImages: boolean;
  disabledTabs: InvokeTabName[];
  disabledFeatures: AppFeature[];
  canRestoreDeletedImagesFromBin: boolean;
  sd: {
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
