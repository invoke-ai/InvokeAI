import { useDisclosure } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { WorkflowLibraryContext } from 'features/nodes/components/flow/WorkflowLibrary/context';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaFolderOpen } from 'react-icons/fa';
import WorkflowLibraryModal from './WorkflowLibraryModal';

const WorkflowLibraryButton = () => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();

  return (
    <WorkflowLibraryContext.Provider value={disclosure}>
      <IAIIconButton
        icon={<FaFolderOpen />}
        onClick={disclosure.onOpen}
        tooltip={t('workflows.workflowLibrary')}
        aria-label={t('workflows.workflowLibrary')}
      />
      <WorkflowLibraryModal />
    </WorkflowLibraryContext.Provider>
  );
};

export default memo(WorkflowLibraryButton);
