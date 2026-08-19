/**
 * Models surface for the Launchpad. Exported eagerly: the Launchpad puts one
 * dynamic-import boundary in front of every live panel at once, so a second
 * one here would only fragment that chunk.
 */
export { ModelsNotice } from './ui/launchpad/ModelsNotice';
/**
 * The Add Models search seam. Exported here rather than from the feature index
 * because the index is a *static* import of the editor bundle — this module is
 * only ever reached by `import()`, so seeding a search from the editor costs
 * the editor nothing until someone clicks.
 */
export { requestAddModelsSearch } from './ui/uiStore';
