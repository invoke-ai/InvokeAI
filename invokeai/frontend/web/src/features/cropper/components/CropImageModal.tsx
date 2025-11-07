import { Modal, ModalBody, ModalContent, ModalHeader, ModalOverlay } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { cropImageModalApi } from 'features/cropper/store';
import { memo } from 'react';

import { CropImageEditor } from './CropImageEditor';

export const CropImageModal = memo(() => {
  const state = useStore(cropImageModalApi.$state);

  if (!state) {
    return null;
  }

  return (
    // This modal is always open when this component is rendered
    <Modal isOpen={true} onClose={cropImageModalApi.close} isCentered useInert={false} size="full">
      <ModalOverlay />
      <ModalContent minH="unset" minW="unset" maxH="90vh" maxW="90vw" w="full" h="full" borderRadius="base">
        <ModalHeader>Crop Image</ModalHeader>
        <ModalBody px={4} pb={4} pt={0}>
          <CropImageEditor editor={state.editor} onApplyCrop={state.onApplyCrop} onReady={state.onReady} />
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

CropImageModal.displayName = 'CropImageModal';
