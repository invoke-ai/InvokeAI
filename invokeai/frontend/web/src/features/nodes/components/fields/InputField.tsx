import { Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import {
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { PropsWithChildren, useMemo } from 'react';
import { NodeProps } from 'reactflow';
import FieldHandle from './FieldHandle';
import FieldTitle from './FieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

interface Props {
  nodeProps: NodeProps<InvocationNodeData>;
  nodeTemplate: InvocationTemplate;
  field: InputFieldValue;
}

const InputField = (props: Props) => {
  const { nodeProps, nodeTemplate, field } = props;
  const { id: nodeId } = nodeProps.data;

  const {
    isConnected,
    isConnectionInProgress,
    isConnectionStartField,
    connectionError,
    shouldDim,
  } = useConnectionState({ nodeId, field, kind: 'input' });

  const fieldTemplate = useMemo(
    () => nodeTemplate.inputs[field.name],
    [field.name, nodeTemplate.inputs]
  );

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

    if (!field.value && !isConnected && fieldTemplate.input === 'any') {
      return true;
    }
  }, [fieldTemplate, isConnected, field.value]);

  if (!fieldTemplate) {
    return (
      <InputFieldWrapper shouldDim={shouldDim}>
        <FormControl
          sx={{ color: 'error.400', textAlign: 'left', fontSize: 'sm' }}
        >
          Unknown input: {field.name}
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
              nodeData={nodeProps.data}
              nodeTemplate={nodeTemplate}
              field={field}
              fieldTemplate={fieldTemplate}
            />
          }
          openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
          placement="top"
          shouldWrapChildren
          hasArrow
        >
          <FormLabel sx={{ mb: 0 }}>
            <FieldTitle
              nodeData={nodeProps.data}
              nodeTemplate={nodeTemplate}
              field={field}
              fieldTemplate={fieldTemplate}
              isDraggable
            />
          </FormLabel>
        </Tooltip>
        <InputFieldRenderer
          nodeData={nodeProps.data}
          nodeTemplate={nodeTemplate}
          field={field}
          fieldTemplate={fieldTemplate}
        />
      </FormControl>

      {fieldTemplate.input !== 'direct' && (
        <FieldHandle
          nodeProps={nodeProps}
          nodeTemplate={nodeTemplate}
          field={field}
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

const InputFieldWrapper = ({ shouldDim, children }: InputFieldWrapperProps) => (
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
);
