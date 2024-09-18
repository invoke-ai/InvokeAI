import {
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import WorkflowLibraryContent from 'features/workflowLibrary/components/WorkflowLibraryContent';
import { $isWorkflowLibraryModalOpen } from 'features/workflowLibrary/store/isWorkflowLibraryModalOpen';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const WorkflowLibraryModal = () => {
  const { t } = useTranslation();
  const isWorkflowLibraryModalOpen = useStore($isWorkflowLibraryModalOpen);

  const onClose = useCallback(() => {
    $isWorkflowLibraryModalOpen.set(false);
  }, []);

  return (
    <Modal isOpen={isWorkflowLibraryModalOpen} onClose={onClose} isCentered useInert={false}>
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
