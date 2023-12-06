import { useDisclosure } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
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
      <IAIButton
        leftIcon={<FaFolderOpen />}
        onClick={disclosure.onOpen}
        pointerEvents="auto"
      >
        {t('workflows.workflowLibrary')}
      </IAIButton>
      <WorkflowLibraryModal />
    </WorkflowLibraryModalContext.Provider>
  );
};

export default memo(WorkflowLibraryButton);
