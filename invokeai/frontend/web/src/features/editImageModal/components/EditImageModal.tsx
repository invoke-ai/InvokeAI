import { Modal, ModalBody, ModalContent, ModalHeader, ModalOverlay } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $editImageModalState, closeEditImageModal } from 'features/editImageModal/store';

import { EditorContainer } from './EditorContainer';

export const EditImageModal = () => {
  const state = useStore($editImageModalState);

  return (
    <Modal isOpen={state.isOpen} onClose={closeEditImageModal} isCentered useInert={false} size="full">
      <ModalOverlay />
      <ModalContent minH="unset" minW="unset" maxH="90vh" maxW="90vw" w="full" h="full" borderRadius="base">
        <ModalHeader>Edit Image</ModalHeader>
        <ModalBody px={4} pb={4} pt={0}>
          {state.isOpen && <EditorContainer editor={state.editor} imageName={state.imageName} />}
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};
