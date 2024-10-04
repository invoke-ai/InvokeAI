import { IconButton } from '@invoke-ai/ui-library';
import { useWorkflowLibraryModal } from 'features/workflowLibrary/store/isWorkflowLibraryModalOpen';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenBold } from 'react-icons/pi';

import WorkflowLibraryModal from './WorkflowLibraryModal';

const WorkflowLibraryButton = () => {
  const { t } = useTranslation();

  const workflowLibraryModal = useWorkflowLibraryModal();

  return (
    <>
      <IconButton
        aria-label={t('workflows.workflowLibrary')}
        tooltip={t('workflows.workflowLibrary')}
        icon={<PiFolderOpenBold />}
        onPointerUp={workflowLibraryModal.setTrue}
        pointerEvents="auto"
      />
      <WorkflowLibraryModal />
    </>
  );
};

export default memo(WorkflowLibraryButton);
