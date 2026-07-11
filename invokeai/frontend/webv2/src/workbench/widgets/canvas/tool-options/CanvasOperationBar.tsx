import type { CanvasOperationState } from '@workbench/canvas-engine/canvasOperationController';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';

import { CanvasOptionsBar } from './CanvasOptionsBar';
import { SamOptions } from './SamOptions';

/** Bottom-center controls for the active guarded operation, independent of temporary tools. */
export const CanvasOperationBar = ({
  engine,
  operation,
}: {
  engine: CanvasEngine;
  operation: Extract<CanvasOperationState, { status: 'active' }>;
}) => (
  <CanvasOptionsBar>
    {operation.identity.kind === 'select-object' ? <SamOptions engine={engine} /> : null}
  </CanvasOptionsBar>
);
