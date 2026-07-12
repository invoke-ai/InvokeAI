/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { CanvasOperationState } from '@workbench/canvas-engine/canvasOperationController';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { ReactNode } from 'react';

import { Box } from '@chakra-ui/react';
import { clearLayerPropertiesRequest } from '@workbench/widgets/layers/layerPropertiesRequestStore';
import { useCallback } from 'react';

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

export const clearLayerPropertiesForOperationPresentation = (): void => clearLayerPropertiesRequest();

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
  const consumeLayerPropertiesRef = useCallback((node: HTMLDivElement | null) => {
    if (node) {
      clearLayerPropertiesForOperationPresentation();
    }
  }, []);
  if (!engine) {
    return null;
  }
  const regularContent = (
    <ToolOptionsBar documentHeight={documentHeight} documentWidth={documentWidth} engine={engine} />
  );
  const renderOperation = (locked: boolean): ReactNode =>
    operation?.status === 'active' ? (
      <Box ref={consumeLayerPropertiesRef} display="contents">
        <CanvasOperationBar engine={engine} isExternalInteractionLocked={locked} operation={operation} />
      </Box>
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
