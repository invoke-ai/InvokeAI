/**
 * Generation's React-free values/policy surface: settings normalization, base
 * policies, batch limits, prompt history, prompt drafts, and reference images.
 * Curated, caller-driven export list — add a symbol only when a consumer needs it.
 */
export {
  ASPECT_RATIO_MAP,
  calculateNewSize,
  clampDimension,
  cloneGenerateWidgetValues,
  DEFAULT_NEGATIVE_PROMPT_HEIGHT_PX,
  DEFAULT_POSITIVE_PROMPT_HEIGHT_PX,
  deriveAspectRatioId,
  GENERATE_UI_STATE_KEYS,
  getDefaultLoraWeight,
  getModelDefaultVae,
  hasModelDefaultVae,
  isLoraCompatibleWithModel,
  isLoraModelConfig,
  isMainModelConfig,
  isModelIdentifierConfig,
  isVaeModelConfig,
  isWanLoraTargetingMain,
  MAX_HIDIFFUSION_RATIO,
  MAX_NEGATIVE_PROMPT_HEIGHT_PX,
  MAX_POSITIVE_PROMPT_HEIGHT_PX,
  MIN_NEGATIVE_PROMPT_HEIGHT_PX,
  MIN_HIDIFFUSION_T1_RATIO,
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
  getGenerateModelSelectionResult,
  getGenerationDimensions,
  getGenerationModelAvailabilityReasons,
  getGenerationUiPolicy,
  getGenerationValidationReasons,
  getMaxReferenceImages,
  getPromptHistoryRecallPatch,
  getSettingsWithModelDefaults,
  isKnownScheduler,
  isReferenceImageSupported,
  isSupportedGenerateModel,
  SCHEDULER_OPTIONS,
  type GenerateModelSelectionResult,
} from './core/baseGenerationPolicies';
export {
  resolveGenerateWidgetValues,
  type ResolvedGenerateWidgetValues,
  type ResolveGenerateWidgetValuesInput,
} from './core/resolveGenerateWidgetValues';
export {
  getCompatibleDiffusersComponentSource,
  isDiffusersMainForBase,
  isVaeCompatibleWithGenerateModel,
  isVaeForBases,
} from './core/componentCompatibility';
export {
  isValidKrea2RebalanceWeights,
  normalizeRebalancePresets,
  type RebalancePreset,
} from './core/conditioningRebalance';
export { MIN_BATCH_COUNT, sanitizeBatchCount } from './core/batch';
export {
  createDynamicPromptsSampleSeed,
  DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS,
  DYNAMIC_PROMPTS_MAX_PROMPTS,
  DYNAMIC_PROMPTS_MIN_PROMPTS,
  hasDynamicPromptSyntax,
  isDynamicPromptsSeedBehaviour,
  sanitizeDynamicPromptsConfig,
  sanitizeMaxPrompts,
  sanitizeSampleSeed,
  type DynamicPromptsConfig,
  type DynamicPromptsSeedBehaviour,
} from './core/dynamicPrompts';
export { getEffectivePrompts } from './core/promptTemplates';
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
