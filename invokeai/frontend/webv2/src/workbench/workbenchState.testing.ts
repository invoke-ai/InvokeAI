/**
 * Reducer-level compatibility facade for invariant tests only.
 *
 * Production code must use Workbench commands and queries. Keeping this facade
 * separate makes reducer coupling visible and lets those tests be retired as
 * command-level coverage replaces them.
 */
export {
  __workbenchReducerInternal as workbenchReducer,
  createInitialWorkbenchState,
  nextLayerName,
} from './workbenchState';
export type { __WorkbenchReducerActionInternal as WorkbenchAction } from './workbenchState';
