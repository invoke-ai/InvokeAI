import {
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
} from '@chakra-ui/react';
import WorkflowLibraryContent from 'features/nodes/components/flow/WorkflowLibrary/WorkflowLibraryContent';
import { useWorkflowLibraryContext } from 'features/nodes/components/flow/WorkflowLibrary/useWorkflowLibraryContext';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const WorkflowLibraryModal = () => {
  const { t } = useTranslation();
  const { isOpen, onClose } = useWorkflowLibraryContext();
  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered>
      <ModalOverlay />
      <ModalContent
        w="80%"
        h="80%"
        minW="unset"
        minH="unset"
        maxW="unset"
        maxH="unset"
      >
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
