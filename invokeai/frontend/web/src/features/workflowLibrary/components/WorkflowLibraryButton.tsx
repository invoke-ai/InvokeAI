import { useDisclosure } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaFolderOpen } from 'react-icons/fa';
import WorkflowLibraryModal from './WorkflowLibraryModal';
import { WorkflowLibraryModalContext } from 'features/workflowLibrary/context/WorkflowLibraryModalContext';

const WorkflowLibraryButton = () => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();

  return (
    <WorkflowLibraryModalContext.Provider value={disclosure}>
      <IAIIconButton
        icon={<FaFolderOpen />}
        onClick={disclosure.onOpen}
        tooltip={t('workflows.workflowLibrary')}
        aria-label={t('workflows.workflowLibrary')}
      />
      <WorkflowLibraryModal />
    </WorkflowLibraryModalContext.Provider>
  );
};

export default memo(WorkflowLibraryButton);
