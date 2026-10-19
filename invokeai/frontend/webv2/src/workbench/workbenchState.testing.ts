/** Test-only reducer facade; production callers use Workbench commands and queries. */
import type { WorkbenchState } from '@workbench/projectContracts';

import {
  createInitialWorkbenchState,
  __workbenchReducerInternal,
  nextLayerName,
  type __WorkbenchReducerActionInternal,
  type WorkbenchReducerContext,
} from './workbenchState';

const DEFAULT_TEST_CONTEXT: WorkbenchReducerContext = { autoSwitchInvocationRoute: true };

export const workbenchReducer = (
  state: WorkbenchState,
  action: __WorkbenchReducerActionInternal,
  context: WorkbenchReducerContext = DEFAULT_TEST_CONTEXT
): WorkbenchState => __workbenchReducerInternal(state, action, context);

export { createInitialWorkbenchState, nextLayerName };
export type { __WorkbenchReducerActionInternal as WorkbenchAction, WorkbenchReducerContext };
