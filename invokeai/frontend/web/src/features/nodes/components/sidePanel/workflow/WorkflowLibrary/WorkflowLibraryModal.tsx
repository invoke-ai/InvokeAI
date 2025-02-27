import {
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';

import { WorkflowLibrarySideNav } from './WorkflowLibrarySideNav';
import { WorkflowLibraryTopNav } from './WorkflowLibraryTopNav';
import { WorkflowList } from './WorkflowList';

export const WorkflowLibraryModal = () => {
  const workflowLibraryModal = useWorkflowLibraryModal();
  return (
    <Modal isOpen={workflowLibraryModal.isOpen} onClose={workflowLibraryModal.close} size="5xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Workflow Library</ModalHeader>
        <ModalCloseButton />
        <ModalBody pb={6}>
          <Flex gap={4}>
            <Flex flexDir="column" gap={4}>
              <WorkflowLibrarySideNav />
            </Flex>
            <Flex flexDir="column" flex={1} gap={6}>
              <WorkflowLibraryTopNav />
              <WorkflowList />
            </Flex>
          </Flex>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};
