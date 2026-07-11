/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { CanvasOperationState } from '@workbench/canvas-engine/canvasOperationController';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { ReactNode } from 'react';

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

export const CanvasBottomControlsPresentation = ({
  isExternalInteractionLocked,
  operationKind,
  regularContent,
  renderOperation,
}: {
  isExternalInteractionLocked: boolean;
  operationKind: 'filter' | 'select-object' | null;
  regularContent: ReactNode;
  renderOperation(isExternalInteractionLocked: boolean): ReactNode;
}) => {
  const slots = resolveBottomControlSlots({ isExternalInteractionLocked, operationKind });
  if (slots.operation) {
    return renderOperation(isExternalInteractionLocked);
  }
  return slots.regular ? regularContent : null;
};

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
  if (!engine) {
    return null;
  }
  const regularContent = (
    <ToolOptionsBar documentHeight={documentHeight} documentWidth={documentWidth} engine={engine} />
  );
  const renderOperation = (locked: boolean): ReactNode =>
    operation?.status === 'active' ? (
      <CanvasOperationBar engine={engine} isExternalInteractionLocked={locked} operation={operation} />
    ) : null;
  return (
    <CanvasBottomControlsPresentation
      isExternalInteractionLocked={isExternalInteractionLocked}
      operationKind={operationKind}
      regularContent={regularContent}
      renderOperation={renderOperation}
    />
  );
};
