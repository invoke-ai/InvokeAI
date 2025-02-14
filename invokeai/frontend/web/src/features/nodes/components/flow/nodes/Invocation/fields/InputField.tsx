import { Flex, FormControl } from '@invoke-ai/ui-library';
import { FieldLinearViewConfigIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/FieldLinearViewConfigIconButton';
import { FieldNotesIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/FieldNotesIconButton';
import FieldResetToDefaultValueButton from 'features/nodes/components/flow/nodes/Invocation/fields/FieldResetToDefaultValueButton';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useFieldInputTemplate } from 'features/nodes/hooks/useFieldInputTemplate';
import { useFieldIsInvalid } from 'features/nodes/hooks/useFieldIsInvalid';
import { memo, useCallback, useState } from 'react';

import EditableFieldTitle from './EditableFieldTitle';
import FieldHandle from './FieldHandle';
import FieldLinearViewToggle from './FieldLinearViewToggle';
import InputFieldRenderer from './InputFieldRenderer';
import { InputFieldWrapper } from './InputFieldWrapper';

interface Props {
  nodeId: string;
  fieldName: string;
}

const InputField = ({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useFieldInputTemplate(nodeId, fieldName);
  const [isHovered, setIsHovered] = useState(false);
  const isInvalid = useFieldIsInvalid(nodeId, fieldName);

  const { isConnected, isConnectionInProgress, isConnectionStartField, validationResult, shouldDim } =
    useConnectionState({ nodeId, fieldName, kind: 'inputs' });

  const onMouseEnter = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseLeave = useCallback(() => {
    setIsHovered(false);
  }, []);

  if (fieldTemplate.input === 'connection' || isConnected) {
    return (
      <InputFieldWrapper shouldDim={shouldDim}>
        <FormControl isInvalid={isInvalid} isDisabled={isConnected} px={2}>
          <EditableFieldTitle
            nodeId={nodeId}
            fieldName={fieldName}
            kind="inputs"
            isInvalid={isInvalid}
            withTooltip
            shouldDim
          />
        </FormControl>

        <FieldHandle
          fieldTemplate={fieldTemplate}
          handleType="target"
          isConnectionInProgress={isConnectionInProgress}
          isConnectionStartField={isConnectionStartField}
          validationResult={validationResult}
        />
      </InputFieldWrapper>
    );
  }

  return (
    <InputFieldWrapper shouldDim={shouldDim}>
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
            <EditableFieldTitle nodeId={nodeId} fieldName={fieldName} kind="inputs" isInvalid={isInvalid} withTooltip />
            {isHovered && <FieldLinearViewConfigIconButton nodeId={nodeId} fieldName={fieldName} />}
            {isHovered && <FieldNotesIconButton nodeId={nodeId} fieldName={fieldName} />}
            {isHovered && <FieldResetToDefaultValueButton nodeId={nodeId} fieldName={fieldName} />}
            {isHovered && <FieldLinearViewToggle nodeId={nodeId} fieldName={fieldName} />}
          </Flex>
          <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
        </Flex>
      </FormControl>

      {fieldTemplate.input !== 'direct' && (
        <FieldHandle
          fieldTemplate={fieldTemplate}
          handleType="target"
          isConnectionInProgress={isConnectionInProgress}
          isConnectionStartField={isConnectionStartField}
          validationResult={validationResult}
        />
      )}
    </InputFieldWrapper>
  );
};

export default memo(InputField);
