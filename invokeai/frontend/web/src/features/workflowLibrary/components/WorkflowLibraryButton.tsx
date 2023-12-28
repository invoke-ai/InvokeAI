import { useDisclosure } from '@chakra-ui/react';
import { InvButton } from 'common/components/InvButton/InvButton';
import { WorkflowLibraryModalContext } from 'features/workflowLibrary/context/WorkflowLibraryModalContext';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaFolderOpen } from 'react-icons/fa';

import WorkflowLibraryModal from './WorkflowLibraryModal';

const WorkflowLibraryButton = () => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();

  return (
    <WorkflowLibraryModalContext.Provider value={disclosure}>
      <InvButton
        leftIcon={<FaFolderOpen />}
        onClick={disclosure.onOpen}
        pointerEvents="auto"
      >
        {t('workflows.workflowLibrary')}
      </InvButton>
      <WorkflowLibraryModal />
    </WorkflowLibraryModalContext.Provider>
  );
};

export default memo(WorkflowLibraryButton);
