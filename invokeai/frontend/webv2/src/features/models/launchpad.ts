/** Export eagerly because Launchpad already shares one lazy boundary across live panels. */
export { ModelsNotice, ModelsRequiredHint } from './ui/launchpad/ModelsNotice';
/** Expose search through this dynamic entry so editor imports do not eagerly load the manager. */
export { requestAddModelsSearch, requestAddModelsTypeFilter } from './ui/uiStore';
