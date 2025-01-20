import { Flex, FormControl } from '@invoke-ai/ui-library';
import { FieldHandle } from 'features/nodes/components/flow/nodes/Invocation/fields/FieldHandle';
import { InputFieldNotesIconButtonEditable } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldNotesIconButtonEditable';
import { InputFieldResetToDefaultValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldResetToDefaultValueIconButton';
import { useInputFieldConnectionState } from 'features/nodes/hooks/useInputFieldConnectionState';
import { useInputFieldIsConnected } from 'features/nodes/hooks/useInputFieldIsConnected';
import { useInputFieldIsInvalid } from 'features/nodes/hooks/useInputFieldIsInvalid';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { memo, useCallback, useState } from 'react';

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

        <FieldHandle
          handleType="target"
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
        isInvalid={isInvalid}
        isDisabled={isConnected}
        // Without pointerEvents prop, disabled inputs don't trigger reactflow events. For example, when making a
        // connection, the mouse up to end the connection won't fire, leaving the connection in-progress.
        pointerEvents={isConnected ? 'none' : 'auto'}
        orientation="vertical"
        px={2}
      >
        <Flex flexDir="column" w="full" gap={1} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
          <Flex gap={1}>
            <InputFieldTitle nodeId={nodeId} fieldName={fieldName} isInvalid={isInvalid} />
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
        <FieldHandle
          handleType="target"
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
