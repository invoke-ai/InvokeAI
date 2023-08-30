import { Box, Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import SelectionOverlay from 'common/components/SelectionOverlay';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useDoesInputHaveValue } from 'features/nodes/hooks/useDoesInputHaveValue';
import { useFieldTemplate } from 'features/nodes/hooks/useFieldTemplate';
import { useIsMouseOverField } from 'features/nodes/hooks/useIsMouseOverField';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { PropsWithChildren, memo, useMemo } from 'react';
import FieldContextMenu from './FieldContextMenu';
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
      <InputFieldWrapper
        nodeId={nodeId}
        fieldName={fieldName}
        shouldDim={shouldDim}
      >
        <FormControl
          sx={{ color: 'error.400', textAlign: 'left', fontSize: 'sm' }}
        >
          Unknown input: {fieldName}
        </FormControl>
      </InputFieldWrapper>
    );
  }

  return (
    <InputFieldWrapper
      nodeId={nodeId}
      fieldName={fieldName}
      shouldDim={shouldDim}
    >
      <FormControl
        isInvalid={isMissingInput}
        isDisabled={isConnected}
        sx={{
          alignItems: 'stretch',
          justifyContent: 'space-between',
          ps: 2,
          gap: 2,
          h: 'full',
          w: 'full',
        }}
      >
        <FieldContextMenu nodeId={nodeId} fieldName={fieldName} kind="input">
          {(ref) => (
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
              hasArrow
            >
              <FormLabel
                sx={{
                  mb: 0,
                  flexShrink: 0,
                  flexGrow: 0,
                }}
              >
                <FieldTitle
                  ref={ref}
                  nodeId={nodeId}
                  fieldName={fieldName}
                  kind="input"
                  isMissingInput={isMissingInput}
                />
              </FormLabel>
            </Tooltip>
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
  nodeId: string;
  fieldName: string;
}>;

const InputFieldWrapper = memo(
  ({ shouldDim, nodeId, fieldName, children }: InputFieldWrapperProps) => {
    const { isMouseOverField, handleMouseOver, handleMouseOut } =
      useIsMouseOverField(nodeId, fieldName);

    return (
      <Flex
        onMouseOver={handleMouseOver}
        onMouseOut={handleMouseOut}
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
        <SelectionOverlay isSelected={false} isHovered={isMouseOverField} />
      </Flex>
    );
  }
);

InputFieldWrapper.displayName = 'InputFieldWrapper';
