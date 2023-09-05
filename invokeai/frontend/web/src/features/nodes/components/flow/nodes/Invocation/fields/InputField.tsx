import { Box, Flex, FormControl, FormLabel } from '@chakra-ui/react';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useDoesInputHaveValue } from 'features/nodes/hooks/useDoesInputHaveValue';
import { useFieldTemplate } from 'features/nodes/hooks/useFieldTemplate';
import { PropsWithChildren, memo, useMemo } from 'react';
import EditableFieldTitle from './EditableFieldTitle';
import FieldContextMenu from './FieldContextMenu';
import FieldHandle from './FieldHandle';
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
        isInvalid={isMissingInput}
        isDisabled={isConnected}
        sx={{
          alignItems: 'stretch',
          justifyContent: 'space-between',
          ps: fieldTemplate.input === 'direct' ? 0 : 2,
          gap: 2,
          h: 'full',
          w: 'full',
        }}
      >
        <FieldContextMenu nodeId={nodeId} fieldName={fieldName} kind="input">
          {(ref) => (
            <FormLabel
              sx={{
                display: 'flex',
                alignItems: 'center',
                h: 'full',
                mb: 0,
                px: 1,
                gap: 2,
              }}
            >
              <EditableFieldTitle
                ref={ref}
                nodeId={nodeId}
                fieldName={fieldName}
                kind="input"
                isMissingInput={isMissingInput}
                withTooltip
              />
            </FormLabel>
          )}
        </FieldContextMenu>
        <Box>
          <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
        </Box>
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

const InputFieldWrapper = memo(
  ({ shouldDim, children }: InputFieldWrapperProps) => {
    return (
      <Flex
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
  }
);

InputFieldWrapper.displayName = 'InputFieldWrapper';
