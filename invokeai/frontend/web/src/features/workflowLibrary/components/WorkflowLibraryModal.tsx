import {
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import WorkflowLibraryContent from 'features/workflowLibrary/components/WorkflowLibraryContent';
import { useWorkflowLibraryModal } from 'features/workflowLibrary/store/isWorkflowLibraryModalOpen';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const WorkflowLibraryModal = () => {
  const { t } = useTranslation();
  const workflowLibraryModal = useWorkflowLibraryModal();

  return (
    <Modal isOpen={workflowLibraryModal.isTrue} onClose={workflowLibraryModal.setFalse} isCentered useInert={false}>
      <ModalOverlay />
      <ModalContent w="80%" h="80%" minW="unset" minH="unset" maxW="1200px" maxH="664px">
        <ModalHeader>{t('workflows.workflowLibrary')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <WorkflowLibraryContent />
        </ModalBody>
        <ModalFooter />
      </ModalContent>
    </Modal>
  );
};

export default memo(WorkflowLibraryModal);
