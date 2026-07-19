/**
 * Generation's React-free values/policy surface: settings normalization, base
 * policies, batch limits, prompt history, prompt drafts, and reference images.
 * Curated, caller-driven export list — add a symbol only when a consumer needs it.
 */
export {
  calculateNewSize,
  clampDimension,
  cloneGenerateWidgetValues,
  DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX,
  DEFAULT_POSITIVE_PROMPT_HEIGHT_PX,
  deriveAspectRatioId,
  getDefaultLoraWeight,
  getModelDefaultVae,
  hasModelDefaultVae,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeModelConfig,
  MAX_NEGATIVE_PROMPT_HEIGHT_PX,
  MAX_POSITIVE_PROMPT_HEIGHT_PX,
  MIN_NEGATIVE_PROMPT_HEIGHT_PX,
  MIN_POSITIVE_PROMPT_HEIGHT_PX,
  normalizeGenerateSettings,
  normalizeGenerateWidgetValues,
  normalizeReferenceImages,
  SEED_MAX,
  syncGenerateWidgetValuesWithModels,
} from './core/settings';
export {
  coerceSchedulerForGraph,
  createReferenceImageId,
  getCompatibleReferenceImages,
  getDefaultGenerateSettings,
  getDefaultReferenceImageConfig,
  getGenerationDimensions,
  getGenerationModelAvailabilityReasons,
  getGenerationUiPolicy,
  getGenerationValidationReasons,
  getMaxReferenceImages,
  getSettingsWithModelDefaults,
  isKnownScheduler,
  isReferenceImageSupported,
  isSupportedGenerateModel,
  SCHEDULER_OPTIONS,
} from './core/baseGenerationPolicies';
export { isVaeCompatibleWithGenerateModel } from './core/componentCompatibility';
export { MIN_BATCH_COUNT, sanitizeBatchCount } from './core/batch';
export {
  addPromptHistoryItem,
  getPromptHistoryItemFromGenerateSettings,
  MAX_PROMPT_HISTORY,
  removePromptHistoryItem,
} from './core/promptHistory';
export {
  applyProjectPromptDraft,
  areProjectPromptDraftsEqual,
  getPromptDraftFromValues,
  migrateProjectPromptDraft,
  type ProjectPromptDraft,
  type ProjectPromptDraftPatch,
} from './core/projectPromptDraft';
export { generatedImageToReferenceImage, getEffectiveReferenceImage } from './core/referenceImage';
