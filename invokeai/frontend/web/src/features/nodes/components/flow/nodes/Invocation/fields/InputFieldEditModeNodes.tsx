import { Flex, FormControl, Spacer } from '@invoke-ai/ui-library';
import { InputFieldDescriptionPopover } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldDescriptionPopover';
import { InputFieldHandle } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldHandle';
import { InputFieldResetToDefaultValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldResetToDefaultValueIconButton';
import { useNodeFieldDnd } from 'features/nodes/components/sidePanel/builder/dnd';
import { useInputFieldConnectionState } from 'features/nodes/hooks/useInputFieldConnectionState';
import { useInputFieldIsConnected } from 'features/nodes/hooks/useInputFieldIsConnected';
import { useInputFieldIsInvalid } from 'features/nodes/hooks/useInputFieldIsInvalid';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { memo, useCallback, useRef, useState } from 'react';

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

  const isDragging = useNodeFieldDnd({ nodeId, fieldName }, fieldTemplate, draggableRef, dragHandleRef);

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
                <InputFieldDescriptionPopover nodeId={nodeId} fieldName={fieldName} />
                <InputFieldResetToDefaultValueIconButton nodeId={nodeId} fieldName={fieldName} />
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
