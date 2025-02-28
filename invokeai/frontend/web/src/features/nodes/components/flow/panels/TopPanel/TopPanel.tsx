import { Flex, IconButton, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import AddNodeButton from 'features/nodes/components/flow/panels/TopPanel/AddNodeButton';
import ClearFlowButton from 'features/nodes/components/flow/panels/TopPanel/ClearFlowButton';
import SaveWorkflowButton from 'features/nodes/components/flow/panels/TopPanel/SaveWorkflowButton';
import UpdateNodesButton from 'features/nodes/components/flow/panels/TopPanel/UpdateNodesButton';
import { useWorkflowEditorSettingsModal } from 'features/nodes/components/flow/panels/TopRightPanel/WorkflowEditorSettings';
import { WorkflowName } from 'features/nodes/components/sidePanel/WorkflowName';
import { selectWorkflowName } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixFill } from 'react-icons/pi';

const TopCenterPanel = () => {
  const name = useAppSelector(selectWorkflowName);
  const modal = useWorkflowEditorSettingsModal();

  const { t } = useTranslation();
  return (
    <Flex gap={2} top={2} left={2} right={2} position="absolute" alignItems="flex-start" pointerEvents="none">
      <Flex gap="2">
        <AddNodeButton />
        <UpdateNodesButton />
      </Flex>
      <Spacer />
      {!!name.length && <WorkflowName />}
      <Spacer />
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
};

export default memo(TopCenterPanel);
