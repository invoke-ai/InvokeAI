import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import ClearFlowButton from 'features/nodes/components/flow/panels/TopPanel/ClearFlowButton';
import SaveWorkflowButton from 'features/nodes/components/flow/panels/TopPanel/SaveWorkflowButton';
import { useWorkflowEditorSettingsModal } from 'features/nodes/components/flow/panels/TopRightPanel/WorkflowEditorSettings';
import { $isInDeployFlow } from 'features/nodes/components/sidePanel/workflow/publish';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixFill } from 'react-icons/pi';

export const TopRightPanel = memo(() => {
  const modal = useWorkflowEditorSettingsModal();
  const isInDeployFlow = useStore($isInDeployFlow);

  const { t } = useTranslation();

  if (isInDeployFlow) {
    return null;
  }

  return (
    <Flex gap={2} top={2} right={2} position="absolute" alignItems="flex-end" pointerEvents="none">
      <ClearFlowButton />
      <SaveWorkflowButton />
      <IconButton
        pointerEvents="auto"
        aria-label={t('workflows.workflowEditorMenu')}
        icon={<PiGearSixFill />}
        onClick={modal.setTrue}
      />
    </Flex>
  );
});

TopRightPanel.displayName = 'TopRightPanel';
