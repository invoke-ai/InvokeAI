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
import { useTranslation } from 'react-i18next';

import { StylePresetForm } from './StylePresetForm';

export const StylePresetModal = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isModalOpen = useAppSelector((s) => s.stylePresetModal.isModalOpen);
  const updatingStylePresetId = useAppSelector((s) => s.stylePresetModal.updatingStylePresetId);

  const modalTitle = useMemo(() => {
    return updatingStylePresetId ? t('stylePresets.updatePromptTemplate') : t('stylePresets.createPromptTemplate');
  }, [updatingStylePresetId, t]);

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
