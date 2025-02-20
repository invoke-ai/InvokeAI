import { Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowName } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const ActiveWorkflowName = memo(() => {
  const workflowName = useAppSelector(selectWorkflowName);

  const { t } = useTranslation();

  if (workflowName) {
    return (
      <Text fontWeight="semibold" fontSize="md" justifySelf="flex-start" noOfLines={1}>
        {workflowName}
      </Text>
    );
  }

  // activeWorkflowName is always a string - if it is an empty string, it implies we do not have a workflow selected
  return (
    <Text fontSize="md" fontWeight="semibold" color="base.300" noOfLines={1}>
      {t('workflows.chooseWorkflowFromLibrary')}
    </Text>
  );
});

ActiveWorkflowName.displayName = 'ActiveWorkflowName';
