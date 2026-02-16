import { Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { linkifyOptions, linkifySx } from 'common/components/linkify';
import { selectWorkflowDescription } from 'features/nodes/store/selectors';
import Linkify from 'linkify-react';
import { memo } from 'react';

export const ActiveWorkflowDescription = memo(() => {
  const description = useAppSelector(selectWorkflowDescription);

  if (!description) {
    return null;
  }

  return (
    <IAITooltip label={description}>
      <Text color="base.300" fontStyle="italic" sx={linkifySx} noOfLines={1}>
        <Linkify options={linkifyOptions}>{description}</Linkify>
      </Text>
    </IAITooltip>
  );
});

ActiveWorkflowDescription.displayName = 'ActiveWorkflowDescription';
