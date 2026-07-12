import type { CanvasOperationState } from '@workbench/canvas-engine/canvasOperationController';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';

import { FilterOptions } from './FilterOptions';
import { SamOptions } from './SamOptions';

/**
 * Bottom-center controls for the active guarded operation, independent of
 * temporary tools. Select Object renders single-row bar chrome (like the
 * per-tool options bars); Filter renders the full operation panel.
 */
export const CanvasOperationBar = ({
  engine,
  isExternalInteractionLocked,
  operation,
}: {
  engine: CanvasEngine;
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
