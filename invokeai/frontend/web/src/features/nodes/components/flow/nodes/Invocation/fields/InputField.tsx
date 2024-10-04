import { Flex, FormControl } from '@invoke-ai/ui-library';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useDoesInputHaveValue } from 'features/nodes/hooks/useDoesInputHaveValue';
import { useFieldInputTemplate } from 'features/nodes/hooks/useFieldInputTemplate';
import { memo, useCallback, useMemo, useState } from 'react';

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
  const doesFieldHaveValue = useDoesInputHaveValue(nodeId, fieldName);
  const [isHovered, setIsHovered] = useState(false);

  const { isConnected, isConnectionInProgress, isConnectionStartField, validationResult, shouldDim } =
    useConnectionState({ nodeId, fieldName, kind: 'inputs' });

  const isMissingInput = useMemo(() => {
    if (!fieldTemplate) {
      return false;
    }

    if (!fieldTemplate.required) {
      return false;
    }

    if (!isConnected && fieldTemplate.input === 'connection') {
      return true;
    }

    if (!doesFieldHaveValue && !isConnected && fieldTemplate.input !== 'connection') {
      return true;
    }

    return false;
  }, [fieldTemplate, isConnected, doesFieldHaveValue]);

  const onMouseEnter = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseLeave = useCallback(() => {
    setIsHovered(false);
  }, []);

  if (fieldTemplate.input === 'connection' || isConnected) {
    return (
      <InputFieldWrapper shouldDim={shouldDim}>
        <FormControl isInvalid={isMissingInput} isDisabled={isConnected} px={2}>
          <EditableFieldTitle
            nodeId={nodeId}
            fieldName={fieldName}
            kind="inputs"
            isMissingInput={isMissingInput}
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
        isInvalid={isMissingInput}
        isDisabled={isConnected}
        // Without pointerEvents prop, disabled inputs don't trigger reactflow events. For example, when making a
        // connection, the mouse up to end the connection won't fire, leaving the connection in-progress.
        pointerEvents={isConnected ? 'none' : 'auto'}
        orientation="vertical"
        px={2}
      >
        <Flex flexDir="column" w="full" gap={1} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
          <Flex>
            <EditableFieldTitle
              nodeId={nodeId}
              fieldName={fieldName}
              kind="inputs"
              isMissingInput={isMissingInput}
              withTooltip
            />
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
