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
import { isModalOpenChanged, updatingStylePresetChanged } from 'features/stylePresets/store/stylePresetModalSlice';
import { useCallback, useMemo } from 'react';

import { StylePresetForm } from './StylePresetForm';

export const StylePresetModal = () => {
  const dispatch = useAppDispatch();
  const isModalOpen = useAppSelector((s) => s.stylePresetModal.isModalOpen);
  const updatingStylePreset = useAppSelector((s) => s.stylePresetModal.updatingStylePreset);

  const modalTitle = useMemo(() => {
    return updatingStylePreset ? `Update Style Preset` : `Create Style Preset`;
  }, [updatingStylePreset]);

  const handleCloseModal = useCallback(() => {
    dispatch(updatingStylePresetChanged(null));
    dispatch(isModalOpenChanged(false));
  }, [dispatch]);

  return (
    <Modal isOpen={isModalOpen} onClose={handleCloseModal} isCentered size="2xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{modalTitle}</ModalHeader>
        <ModalCloseButton />
        <ModalBody display="flex" flexDir="column" gap={4}>
          <StylePresetForm updatingPreset={updatingStylePreset} />
        </ModalBody>
        <ModalFooter />
      </ModalContent>
    </Modal>
  );
};
