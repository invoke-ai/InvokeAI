import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Text, Tooltip } from '@invoke-ai/ui-library';
import { useOutputFieldTemplate } from 'features/nodes/hooks/useOutputFieldTemplate';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

import { OutputFieldTooltip } from './OutputFieldTooltip';

const sx = {
  fontSize: 'sm',
  color: 'base.300',
  fontWeight: 'semibold',
  pe: 2,
  '&[data-is-disabled="true"]': {
    opacity: 0.5,
  },
} satisfies SystemStyleObject;

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
  isDisabled?: boolean;
}>;

export const OutputFieldTitle = memo(({ nodeId, fieldName, isDisabled }: Props) => {
  const fieldTemplate = useOutputFieldTemplate(nodeId, fieldName);

  return (
    <Tooltip
      label={<OutputFieldTooltip nodeId={nodeId} fieldName={fieldName} />}
      openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
      placement="top"
      shouldWrapChildren
    >
      <Text data-is-disabled={isDisabled} sx={sx}>
        {fieldTemplate.title}
      </Text>
    </Tooltip>
  );
});

OutputFieldTitle.displayName = 'OutputFieldTitle';
