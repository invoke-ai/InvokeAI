import {
  Divider,
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { useTranslation } from 'react-i18next';

import { WorkflowLibrarySideNav } from './WorkflowLibrarySideNav';
import { WorkflowLibraryTopNav } from './WorkflowLibraryTopNav';
import { WorkflowList } from './WorkflowList';

export const WorkflowLibraryModal = () => {
  const { t } = useTranslation();
  const workflowLibraryModal = useWorkflowLibraryModal();
  return (
    <Modal isOpen={workflowLibraryModal.isOpen} onClose={workflowLibraryModal.close} isCentered>
      <ModalOverlay />
      <ModalContent
        w="calc(100% - var(--invoke-sizes-40))"
        maxW="calc(100% - var(--invoke-sizes-40))"
        h="calc(100% - var(--invoke-sizes-40))"
        maxH="calc(100% - var(--invoke-sizes-40))"
      >
        <ModalHeader>{t('workflows.workflowLibrary')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody pb={6}>
          <Flex gap={4} h="100%">
            <WorkflowLibrarySideNav />
            <Divider orientation="vertical" />
            <Flex flexDir="column" flex={1} gap={4}>
              <WorkflowLibraryTopNav />
              <WorkflowList />
            </Flex>
          </Flex>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};
