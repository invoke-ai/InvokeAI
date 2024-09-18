import { IconButton } from '@invoke-ai/ui-library';
import { $isWorkflowLibraryModalOpen } from 'features/workflowLibrary/store/isWorkflowLibraryModalOpen';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenBold } from 'react-icons/pi';

import WorkflowLibraryModal from './WorkflowLibraryModal';

const WorkflowLibraryButton = () => {
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    $isWorkflowLibraryModalOpen.set(true);
  }, []);

  return (
    <>
      <IconButton
        aria-label={t('workflows.workflowLibrary')}
        tooltip={t('workflows.workflowLibrary')}
        icon={<PiFolderOpenBold />}
        onClick={onClick}
        pointerEvents="auto"
      />
      <WorkflowLibraryModal />
    </>
  );
};

export default memo(WorkflowLibraryButton);
