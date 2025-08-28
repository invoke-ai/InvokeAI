import { zFilterType } from 'features/controlLayers/store/filters';
import { zParameterPrecision, zParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { zTabName } from 'features/ui/store/uiTypes';
import type { PartialDeep } from 'type-fest';
import z from 'zod';

const zAppFeature = z.enum([
  'faceRestore',
  'upscaling',
  'lightbox',
  'modelManager',
  'githubLink',
  'discordLink',
  'bugLink',
  'aboutModal',
  'localization',
  'consoleLogging',
  'dynamicPrompting',
  'batches',
  'syncModels',
  'multiselect',
  'pauseQueue',
  'resumeQueue',
  'invocationCache',
  'modelCache',
  'bulkDownload',
  'starterModels',
  'hfToken',
  'retryQueueItem',
  'cancelAndClearAll',
  'chatGPT4oHigh',
  'modelRelationships',
]);
export type AppFeature = z.infer<typeof zAppFeature>;

const zSDFeature = z.enum([
  'controlNet',
  'noise',
  'perlinNoise',
  'noiseThreshold',
  'variation',
  'symmetry',
  'seamless',
  'hires',
  'lora',
  'embedding',
  'vae',
  'hrf',
]);
export type SDFeature = z.infer<typeof zSDFeature>;

const zNumericalParameterConfig = z.object({
  initial: z.number().default(512),
  sliderMin: z.number().default(64),
  sliderMax: z.number().default(1536),
  numberInputMin: z.number().default(64),
  numberInputMax: z.number().default(4096),
  fineStep: z.number().default(8),
  coarseStep: z.number().default(64),
});
export type NumericalParameterConfig = z.infer<typeof zNumericalParameterConfig>;

/**
 * Configuration options for the InvokeAI UI.
 * Distinct from system settings which may be changed inside the app.
 */
export const zAppConfig = z.object({
  /**
   * Whether or not we should update image urls when image loading errors
   */
  shouldUpdateImagesOnConnect: z.boolean(),
  shouldFetchMetadataFromApi: z.boolean(),
  /**
   * Sets a size limit for outputs on the upscaling tab. This is a maximum dimension, so the actual max number of pixels
   * will be the square of this value.
   */
  maxUpscaleDimension: z.number().optional(),
  allowPrivateBoards: z.boolean(),
  allowPrivateStylePresets: z.boolean(),
  allowClientSideUpload: z.boolean(),
  allowPublishWorkflows: z.boolean(),
  allowPromptExpansion: z.boolean(),
  disabledTabs: z.array(zTabName),
  disabledFeatures: z.array(zAppFeature),
  disabledSDFeatures: z.array(zSDFeature),
  nodesAllowlist: z.array(z.string()).optional(),
  nodesDenylist: z.array(z.string()).optional(),
  metadataFetchDebounce: z.number().int().optional(),
  workflowFetchDebounce: z.number().int().optional(),
  isLocal: z.boolean().optional(),
  shouldShowCredits: z.boolean().optional(),
  sd: z.object({
    defaultModel: z.string().optional(),
    disabledControlNetModels: z.array(z.string()),
    disabledControlNetProcessors: z.array(zFilterType),
    // Core parameters
    iterations: zNumericalParameterConfig,
    width: zNumericalParameterConfig,
    height: zNumericalParameterConfig,
    steps: zNumericalParameterConfig,
    guidance: zNumericalParameterConfig,
    cfgRescaleMultiplier: zNumericalParameterConfig,
    img2imgStrength: zNumericalParameterConfig,
    scheduler: zParameterScheduler.optional(),
    vaePrecision: zParameterPrecision.optional(),
    // Canvas
    boundingBoxHeight: zNumericalParameterConfig,
    boundingBoxWidth: zNumericalParameterConfig,
    scaledBoundingBoxHeight: zNumericalParameterConfig,
    scaledBoundingBoxWidth: zNumericalParameterConfig,
    canvasCoherenceStrength: zNumericalParameterConfig,
    canvasCoherenceEdgeSize: zNumericalParameterConfig,
    infillTileSize: zNumericalParameterConfig,
    infillPatchmatchDownscaleSize: zNumericalParameterConfig,
    // Misc advanced
    clipSkip: zNumericalParameterConfig, // slider and input max are ignored for this, because the values depend on the model
    maskBlur: zNumericalParameterConfig,
    hrfStrength: zNumericalParameterConfig,
    dynamicPrompts: z.object({
      maxPrompts: zNumericalParameterConfig,
    }),
    ca: z.object({
      weight: zNumericalParameterConfig,
    }),
  }),
  flux: z.object({
    guidance: zNumericalParameterConfig,
  }),
});

export type AppConfig = z.infer<typeof zAppConfig>;
export type PartialAppConfig = PartialDeep<AppConfig>;

export const getDefaultAppConfig = (): AppConfig => ({
  isLocal: true,
  shouldUpdateImagesOnConnect: false,
  shouldFetchMetadataFromApi: false,
  allowPrivateBoards: false,
  allowPrivateStylePresets: false,
  allowClientSideUpload: false,
  allowPublishWorkflows: false,
  allowPromptExpansion: false,
  shouldShowCredits: false,
  disabledTabs: ['video'],
  disabledFeatures: ['lightbox', 'faceRestore', 'batches'] satisfies AppFeature[],
  disabledSDFeatures: ['variation', 'symmetry', 'hires', 'perlinNoise', 'noiseThreshold'] satisfies SDFeature[],
  sd: {
    disabledControlNetModels: [],
    disabledControlNetProcessors: [],
    iterations: {
      initial: 1,
      sliderMin: 1,
      sliderMax: 1000,
      numberInputMin: 1,
      numberInputMax: 10000,
      fineStep: 1,
      coarseStep: 1,
    },
    width: zNumericalParameterConfig.parse({}), // initial value comes from model
    height: zNumericalParameterConfig.parse({}), // initial value comes from model
    boundingBoxWidth: zNumericalParameterConfig.parse({}), // initial value comes from model
    boundingBoxHeight: zNumericalParameterConfig.parse({}), // initial value comes from model
    scaledBoundingBoxWidth: zNumericalParameterConfig.parse({}), // initial value comes from model
    scaledBoundingBoxHeight: zNumericalParameterConfig.parse({}), // initial value comes from model
    scheduler: 'dpmpp_3m_k' as const,
    vaePrecision: 'fp32' as const,
    steps: {
      initial: 30,
      sliderMin: 1,
      sliderMax: 100,
      numberInputMin: 1,
      numberInputMax: 500,
      fineStep: 1,
      coarseStep: 1,
    },
    guidance: {
      initial: 7,
      sliderMin: 1,
      sliderMax: 20,
      numberInputMin: 1,
      numberInputMax: 200,
      fineStep: 0.1,
      coarseStep: 0.5,
    },
    img2imgStrength: {
      initial: 0.7,
      sliderMin: 0,
      sliderMax: 1,
      numberInputMin: 0,
      numberInputMax: 1,
      fineStep: 0.01,
      coarseStep: 0.05,
    },
    canvasCoherenceStrength: {
      initial: 0.3,
      sliderMin: 0,
      sliderMax: 1,
      numberInputMin: 0,
      numberInputMax: 1,
      fineStep: 0.01,
      coarseStep: 0.05,
    },
    hrfStrength: {
      initial: 0.45,
      sliderMin: 0,
      sliderMax: 1,
      numberInputMin: 0,
      numberInputMax: 1,
      fineStep: 0.01,
      coarseStep: 0.05,
    },
    canvasCoherenceEdgeSize: {
      initial: 16,
      sliderMin: 0,
      sliderMax: 128,
      numberInputMin: 0,
      numberInputMax: 1024,
      fineStep: 8,
      coarseStep: 16,
    },
    cfgRescaleMultiplier: {
      initial: 0,
      sliderMin: 0,
      sliderMax: 0.99,
      numberInputMin: 0,
      numberInputMax: 0.99,
      fineStep: 0.05,
      coarseStep: 0.1,
    },
    clipSkip: {
      initial: 0,
      sliderMin: 0,
      sliderMax: 12, // determined by model selection, unused in practice
      numberInputMin: 0,
      numberInputMax: 12, // determined by model selection, unused in practice
      fineStep: 1,
      coarseStep: 1,
    },
    infillPatchmatchDownscaleSize: {
      initial: 1,
      sliderMin: 1,
      sliderMax: 10,
      numberInputMin: 1,
      numberInputMax: 10,
      fineStep: 1,
      coarseStep: 1,
    },
    infillTileSize: {
      initial: 32,
      sliderMin: 16,
      sliderMax: 64,
      numberInputMin: 16,
      numberInputMax: 256,
      fineStep: 1,
      coarseStep: 1,
    },
    maskBlur: {
      initial: 16,
      sliderMin: 0,
      sliderMax: 128,
      numberInputMin: 0,
      numberInputMax: 512,
      fineStep: 1,
      coarseStep: 1,
    },
    ca: {
      weight: {
        initial: 1,
        sliderMin: 0,
        sliderMax: 2,
        numberInputMin: -1,
        numberInputMax: 2,
        fineStep: 0.01,
        coarseStep: 0.05,
      },
    },
    dynamicPrompts: {
      maxPrompts: {
        initial: 100,
        sliderMin: 1,
        sliderMax: 1000,
        numberInputMin: 1,
        numberInputMax: 10000,
        fineStep: 1,
        coarseStep: 10,
      },
    },
  },
  flux: {
    guidance: {
      initial: 4,
      sliderMin: 2,
      sliderMax: 6,
      numberInputMin: 1,
      numberInputMax: 20,
      fineStep: 0.1,
      coarseStep: 0.5,
    },
  },
});
