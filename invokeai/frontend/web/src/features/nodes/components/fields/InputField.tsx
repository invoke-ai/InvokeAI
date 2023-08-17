import { Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import {
  useDoesInputHaveValue,
  useFieldTemplate,
} from 'features/nodes/hooks/useNodeData';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { PropsWithChildren, memo, useMemo } from 'react';
import FieldHandle from './FieldHandle';
import FieldTitle from './FieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

interface Props {
  nodeId: string;
  fieldName: string;
}

const InputField = ({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useFieldTemplate(nodeId, fieldName, 'input');
  const doesFieldHaveValue = useDoesInputHaveValue(nodeId, fieldName);

  const {
    isConnected,
    isConnectionInProgress,
    isConnectionStartField,
    connectionError,
    shouldDim,
  } = useConnectionState({ nodeId, fieldName, kind: 'input' });

  const isMissingInput = useMemo(() => {
    if (fieldTemplate?.fieldKind !== 'input') {
      return false;
    }

    if (!fieldTemplate.required) {
      return false;
    }

    if (!isConnected && fieldTemplate.input === 'connection') {
      return true;
    }

    if (!doesFieldHaveValue && !isConnected && fieldTemplate.input === 'any') {
      return true;
    }
  }, [fieldTemplate, isConnected, doesFieldHaveValue]);

  if (fieldTemplate?.fieldKind !== 'input') {
    return (
      <InputFieldWrapper shouldDim={shouldDim}>
        <FormControl
          sx={{ color: 'error.400', textAlign: 'left', fontSize: 'sm' }}
        >
          Unknown input: {fieldName}
        </FormControl>
      </InputFieldWrapper>
    );
  }

  return (
    <InputFieldWrapper shouldDim={shouldDim}>
      <FormControl
        as={Flex}
        isInvalid={isMissingInput}
        isDisabled={isConnected}
        sx={{
          alignItems: 'center',
          justifyContent: 'space-between',
          ps: 2,
          gap: 2,
        }}
      >
        <Tooltip
          label={
            <FieldTooltipContent
              nodeId={nodeId}
              fieldName={fieldName}
              kind="input"
            />
          }
          openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
          placement="top"
          shouldWrapChildren
          hasArrow
        >
          <FormLabel sx={{ mb: 0 }}>
            <FieldTitle
              nodeId={nodeId}
              fieldName={fieldName}
              kind="input"
              isDraggable
            />
          </FormLabel>
        </Tooltip>
        <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
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

export default InputField;

type InputFieldWrapperProps = PropsWithChildren<{
  shouldDim: boolean;
}>;

const InputFieldWrapper = memo(
  ({ shouldDim, children }: InputFieldWrapperProps) => (
    <Flex
      className="nopan"
      sx={{
        position: 'relative',
        minH: 8,
        py: 0.5,
        alignItems: 'center',
        opacity: shouldDim ? 0.5 : 1,
        transitionProperty: 'opacity',
        transitionDuration: '0.1s',
        w: 'full',
        h: 'full',
      }}
    >
      {children}
    </Flex>
  )
);

InputFieldWrapper.displayName = 'InputFieldWrapper';
