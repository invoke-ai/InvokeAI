import { useDisclosure } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaFolderOpen } from 'react-icons/fa';
import WorkflowLibraryModal from './WorkflowLibraryModal';

const WorkflowLibraryButton = () => {
  const { t } = useTranslation();
  const { isOpen, onClose, onOpen } = useDisclosure();

  return (
    <>
      <IAIIconButton
        icon={<FaFolderOpen />}
        onClick={onOpen}
        tooltip={t('workflows.workflowLibrary')}
        aria-label={t('workflows.workflowLibrary')}
      />
      <WorkflowLibraryModal isOpen={isOpen} onClose={onClose} />
    </>
  );
};

export default memo(WorkflowLibraryButton);
