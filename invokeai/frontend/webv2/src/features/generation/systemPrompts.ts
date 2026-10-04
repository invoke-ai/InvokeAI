/**
 * System prompt read model for other features' pickers. Kept apart from `queries` so consumers that load lazily
 * do not pull the list into the editor's boot graph.
 */
export { systemPromptsQueryOptions, type SystemPromptRecord } from './data/systemPrompts';
