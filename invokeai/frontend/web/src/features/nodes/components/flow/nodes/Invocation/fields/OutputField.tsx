import { Flex } from '@chakra-ui/react';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvLabel } from 'common/components/InvControl/InvLabel';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useFieldOutputInstance } from 'features/nodes/hooks/useFieldOutputInstance';
import { useFieldOutputTemplate } from 'features/nodes/hooks/useFieldOutputTemplate';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';
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
        <InvControl
          alignItems="stretch"
          justifyContent="space-between"
          gap={2}
          h="full"
          w="full"
        >
          <InvLabel
            display="flex"
            alignItems="center"
            h="full"
            color="error.300"
            mb={0}
            px={1}
            gap={2}
          >
            {t('nodes.unknownOutput', {
              name: fieldTemplate?.title ?? fieldName,
            })}
          </InvLabel>
        </InvControl>
      </OutputFieldWrapper>
    );
  }

  return (
    <OutputFieldWrapper shouldDim={shouldDim}>
      <InvTooltip
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
      >
        <InvControl isDisabled={isConnected} pe={2}>
          <InvLabel mb={0}>{fieldTemplate?.title}</InvLabel>
        </InvControl>
      </InvTooltip>
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
      position="relative"
      minH={8}
      py={0.5}
      alignItems="center"
      opacity={shouldDim ? 0.5 : 1}
      transitionProperty="opacity"
      transitionDuration="0.1s"
      justifyContent="flex-end"
    >
      {children}
    </Flex>
  )
);

OutputFieldWrapper.displayName = 'OutputFieldWrapper';
