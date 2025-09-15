import { Modal, ModalBody, ModalContent, ModalHeader, ModalOverlay } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $editImageModalState, closeEditImageModal } from 'features/editImageModal/store';

import { EditorContainer } from './EditorContainer';

export const EditImageModal = () => {
  const state = useStore($editImageModalState);

  if (!state) {
    return null;
  }

  return (
    <Modal isOpen={true} onClose={closeEditImageModal} isCentered useInert={false} size="full">
      <ModalOverlay />
      <ModalContent minH="unset" minW="unset" maxH="90vh" maxW="90vw" w="full" h="full" borderRadius="base">
        <ModalHeader>Crop Image</ModalHeader>
        <ModalBody px={4} pb={4} pt={0}>
          <EditorContainer editor={state.editor} onApplyCrop={state.onApplyCrop} onReady={state.onReady} />
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};
