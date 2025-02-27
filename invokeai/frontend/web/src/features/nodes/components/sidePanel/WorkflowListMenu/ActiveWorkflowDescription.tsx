import { Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowDescription } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';

export const ActiveWorkflowDescription = memo(() => {
  const description = useAppSelector(selectWorkflowDescription);

  if (!description) {
    return null;
  }

  return (
    <Text color="base.300" fontStyle="italic" noOfLines={1} pb={2}>
      {description}
    </Text>
  );
});

ActiveWorkflowDescription.displayName = 'ActiveWorkflowDescription';
