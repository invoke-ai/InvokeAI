/**
 * Generation's React integration surface: the UI port provider, draft
 * registry/flushing, form view-model selectors, and prompt hotkey handlers.
 * Curated, caller-driven export list — add a symbol only when a consumer needs it.
 */
export { GenerationUiProvider, type GenerationUiAdapter } from './ui/GenerationUiContext';
export { flushGenerateDrafts, useRegisterGenerateDraftFlusher } from './ui/generateDraftRegistry';
export { useDebouncedDraftValue } from './ui/useDebouncedDraftValue';
export { createGenerateFormValuesSelector } from './ui/generateFormViewModel';
export { adjustFocusedPromptAttention } from './ui/promptFields/promptAttentionHotkeys';
export { focusPositivePrompt } from './ui/promptFields/promptFocus';
export { promptHistoryNavigation } from './ui/promptFields/promptHistoryNavigation';
