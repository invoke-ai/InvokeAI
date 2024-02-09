import { Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useDoesInputHaveValue } from 'features/nodes/hooks/useDoesInputHaveValue';
import { useFieldInputInstance } from 'features/nodes/hooks/useFieldInputInstance';
import { useFieldInputTemplate } from 'features/nodes/hooks/useFieldInputTemplate';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import EditableFieldTitle from './EditableFieldTitle';
import FieldContextMenu from './FieldContextMenu';
import FieldHandle from './FieldHandle';
import InputFieldRenderer from './InputFieldRenderer';

interface Props {
  nodeId: string;
  fieldName: string;
}

const InputField = ({ nodeId, fieldName }: Props) => {
  const { t } = useTranslation();
  const fieldTemplate = useFieldInputTemplate(nodeId, fieldName);
  const fieldInstance = useFieldInputInstance(nodeId, fieldName);
  const doesFieldHaveValue = useDoesInputHaveValue(nodeId, fieldName);

  const { isConnected, isConnectionInProgress, isConnectionStartField, connectionError, shouldDim } =
    useConnectionState({ nodeId, fieldName, kind: 'input' });

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

  if (!fieldTemplate || !fieldInstance) {
    return (
      <InputFieldWrapper shouldDim={shouldDim}>
        <FormControl alignItems="stretch" justifyContent="space-between" flexDir="column" gap={2} h="full" w="full">
          <FormLabel display="flex" alignItems="center" mb={0} px={1} gap={2} h="full">
            {t('nodes.unknownInput', {
              name: fieldInstance?.label ?? fieldTemplate?.title ?? fieldName,
            })}
          </FormLabel>
        </FormControl>
      </InputFieldWrapper>
    );
  }

  if (fieldTemplate.input === 'connection') {
    return (
      <InputFieldWrapper shouldDim={shouldDim}>
        <FormControl isInvalid={isMissingInput} isDisabled={isConnected} px={2}>
          <EditableFieldTitle
            nodeId={nodeId}
            fieldName={fieldName}
            kind="input"
            isMissingInput={isMissingInput}
            withTooltip
          />
        </FormControl>

        <FieldHandle
          fieldTemplate={fieldTemplate}
          handleType="target"
          isConnectionInProgress={isConnectionInProgress}
          isConnectionStartField={isConnectionStartField}
          connectionError={connectionError}
        />
      </InputFieldWrapper>
    );
  }

  return (
    <InputFieldWrapper shouldDim={shouldDim}>
      <FormControl isInvalid={isMissingInput} isDisabled={isConnected} orientation="vertical" px={2}>
        <Flex flexDir="column" w="full" gap={1}>
          <FieldContextMenu nodeId={nodeId} fieldName={fieldName} kind="input">
            {(ref) => (
              <EditableFieldTitle
                ref={ref}
                nodeId={nodeId}
                fieldName={fieldName}
                kind="input"
                isMissingInput={isMissingInput}
                withTooltip
              />
            )}
          </FieldContextMenu>
          <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
        </Flex>
      </FormControl>

      {fieldTemplate.input !== 'direct' && (
        <FieldHandle
          fieldTemplate={fieldTemplate}
          handleType="target"
          isConnectionInProgress={isConnectionInProgress}
          isConnectionStartField={isConnectionStartField}
          connectionError={connectionError}
        />
      )}
    </InputFieldWrapper>
  );
};

export default memo(InputField);

type InputFieldWrapperProps = PropsWithChildren<{
  shouldDim: boolean;
}>;

const InputFieldWrapper = memo(({ shouldDim, children }: InputFieldWrapperProps) => {
  return (
    <Flex
      position="relative"
      minH={8}
      py={0.5}
      alignItems="center"
      opacity={shouldDim ? 0.5 : 1}
      transitionProperty="opacity"
      transitionDuration="0.1s"
      w="full"
      h="full"
    >
      {children}
    </Flex>
  );
});

InputFieldWrapper.displayName = 'InputFieldWrapper';
