import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Flex, FormControl, Spacer } from '@invoke-ai/ui-library';
import { firefoxDndFix } from 'features/dnd/util';
import { InputFieldHandle } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldHandle';
import { InputFieldNotesIconButtonEditable } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldNotesIconButtonEditable';
import { InputFieldResetToDefaultValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldResetToDefaultValueIconButton';
import { buildNodeFieldDndData } from 'features/nodes/components/sidePanel/builder/use-builder-dnd';
import { useInputFieldConnectionState } from 'features/nodes/hooks/useInputFieldConnectionState';
import { useInputFieldIsConnected } from 'features/nodes/hooks/useInputFieldIsConnected';
import { useInputFieldIsInvalid } from 'features/nodes/hooks/useInputFieldIsInvalid';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import type { FieldIdentifier } from 'features/nodes/types/field';
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';

import { InputFieldAddRemoveLinearViewIconButton } from './InputFieldAddRemoveLinearViewIconButton';
import { InputFieldRenderer } from './InputFieldRenderer';
import { InputFieldTitle } from './InputFieldTitle';
import { InputFieldWrapper } from './InputFieldWrapper';

interface Props {
  nodeId: string;
  fieldName: string;
}

export const InputFieldEditModeNodes = memo(({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useInputFieldTemplate(nodeId, fieldName);
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const [isHovered, setIsHovered] = useState(false);
  const isInvalid = useInputFieldIsInvalid(nodeId, fieldName);
  const isConnected = useInputFieldIsConnected(nodeId, fieldName);
  const { isConnectionInProgress, isConnectionStartField, validationResult } = useInputFieldConnectionState(
    nodeId,
    fieldName
  );

  const onMouseEnter = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseLeave = useCallback(() => {
    setIsHovered(false);
  }, []);

  const isDragging = useNodeFieldDnd({ nodeId, fieldName }, draggableRef, dragHandleRef);

  if (fieldTemplate.input === 'connection' || isConnected) {
    return (
      <InputFieldWrapper>
        <FormControl isInvalid={isInvalid} isDisabled={isConnected} px={2}>
          <InputFieldTitle
            nodeId={nodeId}
            fieldName={fieldName}
            isInvalid={isInvalid}
            isDisabled={(isConnectionInProgress && !validationResult.isValid && !isConnectionStartField) || isConnected}
          />
        </FormControl>

        <InputFieldHandle
          fieldTemplate={fieldTemplate}
          isConnectionInProgress={isConnectionInProgress}
          isConnectionStartField={isConnectionStartField}
          validationResult={validationResult}
        />
      </InputFieldWrapper>
    );
  }

  return (
    <InputFieldWrapper>
      <FormControl
        ref={draggableRef}
        isInvalid={isInvalid}
        isDisabled={isConnected}
        // Without pointerEvents prop, disabled inputs don't trigger reactflow events. For example, when making a
        // connection, the mouse up to end the connection won't fire, leaving the connection in-progress.
        pointerEvents={isConnected ? 'none' : 'auto'}
        orientation="vertical"
        px={2}
        opacity={isDragging ? 0.3 : 1}
      >
        <Flex flexDir="column" w="full" gap={1} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
          <Flex className="nodrag" ref={dragHandleRef} gap={1}>
            <InputFieldTitle nodeId={nodeId} fieldName={fieldName} isInvalid={isInvalid} />
            <Spacer />
            {isHovered && (
              <>
                <InputFieldNotesIconButtonEditable nodeId={nodeId} fieldName={fieldName} />
                <InputFieldResetToDefaultValueIconButton nodeId={nodeId} fieldName={fieldName} />
                <InputFieldAddRemoveLinearViewIconButton nodeId={nodeId} fieldName={fieldName} />
              </>
            )}
          </Flex>
          <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
        </Flex>
      </FormControl>

      {fieldTemplate.input !== 'direct' && (
        <InputFieldHandle
          fieldTemplate={fieldTemplate}
          isConnectionInProgress={isConnectionInProgress}
          isConnectionStartField={isConnectionStartField}
          validationResult={validationResult}
        />
      )}
    </InputFieldWrapper>
  );
});

InputFieldEditModeNodes.displayName = 'InputFieldEditModeNodes';

const useNodeFieldDnd = (
  fieldIdentifier: FieldIdentifier,
  draggableRef: RefObject<HTMLElement>,
  dragHandleRef: RefObject<HTMLElement>
) => {
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const draggableElement = draggableRef.current;
    const dragHandleElement = dragHandleRef.current;
    if (!draggableElement || !dragHandleElement) {
      return;
    }
    return combine(
      firefoxDndFix(draggableElement),
      draggable({
        element: draggableElement,
        dragHandle: dragHandleElement,
        getInitialData: () => buildNodeFieldDndData(fieldIdentifier),
        onDragStart: () => {
          setIsDragging(true);
        },
        onDrop: () => {
          setIsDragging(false);
        },
      })
    );
  }, [dragHandleRef, draggableRef, fieldIdentifier]);

  return isDragging;
};
