/** Test-only reducer facade; production callers use Workbench commands and queries. */
import type { WorkbenchState } from '@workbench/projectContracts';

import { captureInvocationRuntimeInputs } from './invocationPreparation';
import {
  createInitialWorkbenchState,
  __workbenchReducerInternal,
  nextLayerName,
  type __WorkbenchReducerActionInternal,
  type WorkbenchReducerContext,
} from './workbenchState';

type Submission = Extract<__WorkbenchReducerActionInternal, { runtime: unknown }>;
type OptionalRuntime<T> = T extends Submission ? Omit<T, 'runtime'> & { runtime?: T['runtime'] } : T;
type TestAction = OptionalRuntime<__WorkbenchReducerActionInternal>;

const DEFAULT_TEST_CONTEXT: WorkbenchReducerContext = { autoSwitchInvocationRoute: true };

export const workbenchReducer = (
  state: WorkbenchState,
  action: TestAction,
  context: WorkbenchReducerContext = DEFAULT_TEST_CONTEXT
): WorkbenchState =>
  __workbenchReducerInternal(
    state,
    action.type === 'submitResolvedInvocationSnapshot' ||
      action.type === 'submitCanvasInvocationSnapshot' ||
      action.type === 'submitModelOnlyUpscale'
      ? ({
          ...action,
          runtime:
            action.runtime ??
            captureInvocationRuntimeInputs(
              action.type === 'submitResolvedInvocationSnapshot'
                ? action.route.sourceId
                : action.type === 'submitCanvasInvocationSnapshot'
                  ? 'canvas'
                  : 'upscale'
            ),
        } as __WorkbenchReducerActionInternal)
      : (action as __WorkbenchReducerActionInternal),
    context
  );

export { createInitialWorkbenchState, nextLayerName };
export type { TestAction as WorkbenchAction, WorkbenchReducerContext };
