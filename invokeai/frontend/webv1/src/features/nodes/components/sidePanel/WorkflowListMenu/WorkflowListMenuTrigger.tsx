import { Button, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowName } from 'features/nodes/store/selectors';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenFill } from 'react-icons/pi';

export const WorkflowListMenuTrigger = () => {
  const workflowLibraryModal = useWorkflowLibraryModal();
  const { t } = useTranslation();
  const workflowName = useAppSelector(selectWorkflowName);

  return (
    <Button variant="ghost" rightIcon={<PiFolderOpenFill />} size="sm" onClick={workflowLibraryModal.open}>
      <Text
        display="auto"
        noOfLines={1}
        overflow="hidden"
        textOverflow="ellipsis"
        whiteSpace="nowrap"
        wordBreak="break-all"
      >
        {workflowName || t('workflows.chooseWorkflowFromLibrary')}
      </Text>
    </Button>
  );
};
