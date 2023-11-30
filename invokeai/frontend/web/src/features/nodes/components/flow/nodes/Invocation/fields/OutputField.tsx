import { Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useFieldOutputInstance } from 'features/nodes/hooks/useFieldOutputInstance';
import { useFieldOutputTemplate } from 'features/nodes/hooks/useFieldOutputTemplate';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { PropsWithChildren, memo } from 'react';
import { useTranslation } from 'react-i18next';
import FieldHandle from './FieldHandle';
import FieldTooltipContent from './FieldTooltipContent';

interface Props {
  nodeId: string;
  fieldName: string;
}

const OutputField = ({ nodeId, fieldName }: Props) => {
  const { t } = useTranslation();
  const fieldTemplate = useFieldOutputTemplate(nodeId, fieldName);
  const fieldInstance = useFieldOutputInstance(nodeId, fieldName);

  const {
    isConnected,
    isConnectionInProgress,
    isConnectionStartField,
    connectionError,
    shouldDim,
  } = useConnectionState({ nodeId, fieldName, kind: 'output' });

  if (!fieldTemplate || !fieldInstance) {
    return (
      <OutputFieldWrapper shouldDim={shouldDim}>
        <FormControl
          sx={{
            alignItems: 'stretch',
            justifyContent: 'space-between',
            gap: 2,
            h: 'full',
            w: 'full',
          }}
        >
          <FormLabel
            sx={{
              display: 'flex',
              alignItems: 'center',
              mb: 0,
              px: 1,
              gap: 2,
              h: 'full',
              fontWeight: 600,
              color: 'error.400',
              _dark: { color: 'error.300' },
            }}
          >
            {t('nodes.unknownOutput', {
              name: fieldTemplate?.title ?? fieldName,
            })}
          </FormLabel>
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
