import { Modal, ModalBody, ModalContent, ModalHeader, ModalOverlay } from "@invoke-ai/ui-library";
import { useStore } from "@nanostores/react";
import { $isOpen } from "features/editImageModal/store";
import { useCallback } from "react";

import { EditorContainer } from "./EditorContainer";

export const EditImageModal = () => {
  const isOpen = useStore($isOpen);
  const onClose = useCallback(() => {
    $isOpen.set(false);
  }, []);

  return <Modal
    isOpen={isOpen}
    onClose={onClose}
    isCentered
    useInert={false}
    size="6xl"
  >
    <ModalOverlay />
    <ModalContent h="80vh">
      <ModalHeader>Edit Image</ModalHeader>
      <ModalBody >
        <EditorContainer />
      </ModalBody>
    </ModalContent>
  </Modal>;
};