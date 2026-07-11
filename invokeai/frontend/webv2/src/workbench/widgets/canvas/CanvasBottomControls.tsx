import type { CanvasOperationState } from '@workbench/canvas-engine/canvasOperationController';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';

import { CanvasOperationBar } from './tool-options/CanvasOperationBar';
import { ToolOptionsBar } from './tool-options/ToolOptionsBar';

export const resolveBottomControlSlots = ({
  isExternalInteractionLocked,
  operationKind,
}: {
  isExternalInteractionLocked: boolean;
  operationKind: 'filter' | 'select-object' | null;
}): { operation: boolean; regular: boolean } => ({
  operation: operationKind !== null,
  regular: operationKind === null && !isExternalInteractionLocked,
});

export const CanvasBottomControls = ({
  documentHeight,
  documentWidth,
  engine,
  isExternalInteractionLocked,
  operation,
}: {
  documentHeight: number | null;
  documentWidth: number | null;
  engine: CanvasEngine | null;
  isExternalInteractionLocked: boolean;
  operation: CanvasOperationState | null;
}) => {
  const operationKind = operation?.status === 'active' ? operation.identity.kind : null;
  const slots = resolveBottomControlSlots({ isExternalInteractionLocked, operationKind });
  if (!engine) {
    return null;
  }
  if (slots.operation && operation?.status === 'active') {
    return (
      <CanvasOperationBar
        engine={engine}
        isExternalInteractionLocked={isExternalInteractionLocked}
        operation={operation}
      />
    );
  }
  if (slots.regular) {
    return <ToolOptionsBar documentHeight={documentHeight} documentWidth={documentWidth} engine={engine} />;
  }
  return null;
};
