import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Text, Tooltip } from '@invoke-ai/ui-library';
import { OutputFieldTooltipContent } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldTooltipContent';
import {
  useConnectionErrorTKey,
  useIsConnectionInProgress,
  useIsConnectionStartField,
} from 'features/nodes/hooks/useFieldConnectionState';
import { useInputFieldIsConnected } from 'features/nodes/hooks/useInputFieldIsConnected';
import { useOutputFieldTemplate } from 'features/nodes/hooks/useOutputFieldTemplate';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo } from 'react';

const sx = {
  fontSize: 'sm',
  color: 'base.300',
  fontWeight: 'semibold',
  pe: 2,
  '&[data-is-disabled="true"]': {
    opacity: 0.5,
  },
} satisfies SystemStyleObject;

type Props = {
  nodeId: string;
  fieldName: string;
};

export const OutputFieldTitle = memo(({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useOutputFieldTemplate(nodeId, fieldName);
  const isConnected = useInputFieldIsConnected(nodeId, fieldName);
  const isConnectionStartField = useIsConnectionStartField(nodeId, fieldName, 'source');
  const isConnectionInProgress = useIsConnectionInProgress();
  const connectionErrorTKey = useConnectionErrorTKey(nodeId, fieldName, 'source');

  return (
    <Tooltip
      label={<OutputFieldTooltipContent nodeId={nodeId} fieldName={fieldName} />}
      openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
      placement="top"
    >
      <Text
        data-is-disabled={
          (isConnectionInProgress && connectionErrorTKey !== null && !isConnectionStartField) || isConnected
        }
        sx={sx}
      >
        {fieldTemplate.title}
      </Text>
    </Tooltip>
  );
});

OutputFieldTitle.displayName = 'OutputFieldTitle';
