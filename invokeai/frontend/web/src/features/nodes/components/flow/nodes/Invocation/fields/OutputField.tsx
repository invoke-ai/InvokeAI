import { Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useFieldTemplate } from 'features/nodes/hooks/useFieldTemplate';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { PropsWithChildren, memo } from 'react';
import FieldHandle from './FieldHandle';
import FieldTooltipContent from './FieldTooltipContent';

interface Props {
  nodeId: string;
  fieldName: string;
}

const OutputField = ({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useFieldTemplate(nodeId, fieldName, 'output');

  const {
    isConnected,
    isConnectionInProgress,
    isConnectionStartField,
    connectionError,
    shouldDim,
  } = useConnectionState({ nodeId, fieldName, kind: 'output' });

  if (fieldTemplate?.fieldKind !== 'output') {
    return (
      <OutputFieldWrapper shouldDim={shouldDim}>
        <FormControl
          sx={{ color: 'error.400', textAlign: 'right', fontSize: 'sm' }}
        >
          Unknown output: {fieldName}
        </FormControl>
      </OutputFieldWrapper>
    );
  }

  return (
    <OutputFieldWrapper shouldDim={shouldDim}>
      <Tooltip
        label={
          <FieldTooltipContent
            nodeId={nodeId}
            fieldName={fieldName}
            kind="output"
          />
        }
        openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
        placement="top"
        shouldWrapChildren
        hasArrow
      >
        <FormControl isDisabled={isConnected} pe={2}>
          <FormLabel sx={{ mb: 0, fontWeight: 500 }}>
            {fieldTemplate?.title}
          </FormLabel>
        </FormControl>
      </Tooltip>
      <FieldHandle
        fieldTemplate={fieldTemplate}
        handleType="source"
        isConnectionInProgress={isConnectionInProgress}
        isConnectionStartField={isConnectionStartField}
        connectionError={connectionError}
      />
    </OutputFieldWrapper>
  );
};

export default memo(OutputField);

type OutputFieldWrapperProps = PropsWithChildren<{
  shouldDim: boolean;
}>;

const OutputFieldWrapper = memo(
  ({ shouldDim, children }: OutputFieldWrapperProps) => (
    <Flex
      sx={{
        position: 'relative',
        minH: 8,
        py: 0.5,
        alignItems: 'center',
        opacity: shouldDim ? 0.5 : 1,
        transitionProperty: 'opacity',
        transitionDuration: '0.1s',
        justifyContent: 'flex-end',
      }}
    >
      {children}
    </Flex>
  )
);

OutputFieldWrapper.displayName = 'OutputFieldWrapper';
