import { Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { linkifyOptions, linkifySx } from 'common/components/linkify';
import { selectWorkflowDescription } from 'features/nodes/store/workflowSlice';
import Linkify from 'linkify-react';
import { memo } from 'react';

export const ActiveWorkflowDescription = memo(() => {
  const description = useAppSelector(selectWorkflowDescription);

  if (!description) {
    return null;
  }

  return (
    <Text color="base.300" fontStyle="italic" pb={2} sx={linkifySx}>
      <Linkify options={linkifyOptions}>{description}</Linkify>
    </Text>
  );
});

ActiveWorkflowDescription.displayName = 'ActiveWorkflowDescription';
