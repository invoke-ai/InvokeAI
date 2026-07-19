import type { CanvasOperationState } from '@workbench/canvas-operations/api';

import type { CanvasOperationUIEngine } from './operationUIEngine';

import { FilterOptions } from './FilterOptions';
import { SamOptions } from './SamOptions';

/**
 * Bottom-center controls for the active guarded operation, independent of
 * temporary tools. Both operations render single-row floating-bar chrome,
 * matching the per-tool options bars.
 */
export const CanvasOperationBar = ({
  engine,
  isExternalInteractionLocked,
  operation,
}: {
  engine: CanvasOperationUIEngine;
  isExternalInteractionLocked: boolean;
  operation: Extract<CanvasOperationState, { status: 'active' }>;
}) => (
  <>
    {operation.identity.kind === 'select-object' ? (
      <SamOptions engine={engine} isExternalInteractionLocked={isExternalInteractionLocked} />
    ) : null}
    {operation.identity.kind === 'filter' ? (
      <FilterOptions engine={engine} isExternalInteractionLocked={isExternalInteractionLocked} />
    ) : null}
  </>
);
