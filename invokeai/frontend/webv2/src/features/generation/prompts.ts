/**
 * Generation's prompt-expansion transport surface: the shared dynamic prompts
 * cache used by the Generate preview, the Invoke tooltip, and Queue's
 * submission. The pure policy (`hasDynamicPromptSyntax`,
 * `buildGeneratePromptBatchPlan`, bounds and config sanitization) lives on
 * `@features/generation/settings`.
 * Curated, caller-driven export list — add a symbol only when a consumer needs it.
 */
export { dynamicPromptsKeys, dynamicPromptsQueryOptions, resolveDynamicPrompts } from './data/dynamicPromptsQueries';
export type { ParseDynamicPromptsRequest, ParseDynamicPromptsResponse } from './data/promptUtilities';
