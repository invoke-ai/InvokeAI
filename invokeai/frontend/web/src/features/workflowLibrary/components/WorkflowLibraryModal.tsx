import {
  InvModal,
  InvModalBody,
  InvModalCloseButton,
  InvModalContent,
  InvModalFooter,
  InvModalHeader,
  InvModalOverlay,
} from 'common/components/InvModal/wrapper';
import WorkflowLibraryContent from 'features/workflowLibrary/components/WorkflowLibraryContent';
import { useWorkflowLibraryModalContext } from 'features/workflowLibrary/context/useWorkflowLibraryModalContext';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const WorkflowLibraryModal = () => {
  const { t } = useTranslation();
  const { isOpen, onClose } = useWorkflowLibraryModalContext();
  return (
    <InvModal isOpen={isOpen} onClose={onClose} isCentered>
      <InvModalOverlay />
      <InvModalContent
        w="80%"
        h="80%"
        minW="unset"
        minH="unset"
        maxW="1200px"
        maxH="664px"
      >
        <InvModalHeader>{t('workflows.workflowLibrary')}</InvModalHeader>
        <InvModalCloseButton />
        <InvModalBody>
          <WorkflowLibraryContent />
        </InvModalBody>
        <InvModalFooter />
      </InvModalContent>
    </InvModal>
  );
};

export default memo(WorkflowLibraryModal);
