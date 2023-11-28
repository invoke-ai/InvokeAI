import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
} from '@chakra-ui/react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import WorkflowLibraryWrapper from './WorkflowLibraryWrapper';

type Props = {
  isOpen: boolean;
  onClose: () => void;
};

const WorkflowLibraryModal = ({ isOpen, onClose }: Props) => {
  const { t } = useTranslation();
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
          <WorkflowLibraryWrapper />
        </ModalBody>
        <ModalFooter />
      </ModalContent>
    </Modal>
  );
};

export default memo(WorkflowLibraryModal);
