import { Button } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { $isInDeployFlow } from 'features/nodes/components/sidePanel/builder/deploy';
import { selectIsWorkflowSaved } from 'features/nodes/store/workflowSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightningFill } from 'react-icons/pi';

export const DeployWorkflowButton = memo(() => {
  const { t } = useTranslation();
  const deployWorkflowIsEnabled = useFeatureStatus('deployWorkflow');
  const isWorkflowSaved = useAppSelector(selectIsWorkflowSaved);

  const onClick = useCallback(() => {
    $isInDeployFlow.set(true);
  }, []);

  return (
    <Button
      onClick={onClick}
      leftIcon={<PiLightningFill />}
      variant="ghost"
      size="sm"
      isDisabled={!deployWorkflowIsEnabled || !isWorkflowSaved}
    >
      {t('workflows.builder.publish')}
    </Button>
  );
});

DeployWorkflowButton.displayName = 'DeployWorkflowButton';
