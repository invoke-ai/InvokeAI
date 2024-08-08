import {
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isModalOpenChanged, updatingStylePresetIdChanged } from 'features/stylePresets/store/stylePresetModalSlice';
import { useCallback, useMemo } from 'react';

import { StylePresetForm } from './StylePresetForm';

export const StylePresetModal = () => {
  const dispatch = useAppDispatch();
  const isModalOpen = useAppSelector((s) => s.stylePresetModal.isModalOpen);
  const updatingStylePresetId = useAppSelector((s) => s.stylePresetModal.updatingStylePresetId);

  const modalTitle = useMemo(() => {
    return updatingStylePresetId ? `Update Style Preset` : `Create Style Preset`;
  }, [updatingStylePresetId]);

  const handleCloseModal = useCallback(() => {
    dispatch(updatingStylePresetIdChanged(null));
    dispatch(isModalOpenChanged(false));
  }, [dispatch]);

  return (
    <Modal isOpen={isModalOpen} onClose={handleCloseModal} isCentered size="2xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{modalTitle}</ModalHeader>
        <ModalCloseButton />
        <ModalBody display="flex" flexDir="column" gap={4}>
          <StylePresetForm updatingStylePresetId={updatingStylePresetId} />
        </ModalBody>
        <ModalFooter />
      </ModalContent>
    </Modal>
  );
};
