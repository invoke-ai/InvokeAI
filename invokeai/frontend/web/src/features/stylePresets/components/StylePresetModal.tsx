import {
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Spinner,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { convertImageUrlToBlob } from 'common/util/convertImageUrlToBlob';
import {
  isModalOpenChanged,
  prefilledFormDataChanged,
  updatingStylePresetIdChanged,
} from 'features/stylePresets/store/stylePresetModalSlice';
import type { PrefilledFormData } from 'features/stylePresets/store/types';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { StylePresetFormData } from './StylePresetForm';
import { StylePresetForm } from './StylePresetForm';

export const StylePresetModal = () => {
  const [formData, setFormData] = useState<StylePresetFormData | null>(null);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isModalOpen = useAppSelector((s) => s.stylePresetModal.isModalOpen);
  const updatingStylePresetId = useAppSelector((s) => s.stylePresetModal.updatingStylePresetId);
  const prefilledFormData = useAppSelector((s) => s.stylePresetModal.prefilledFormData);

  const modalTitle = useMemo(() => {
    return updatingStylePresetId ? t('stylePresets.updatePromptTemplate') : t('stylePresets.createPromptTemplate');
  }, [updatingStylePresetId, t]);

  const handleCloseModal = useCallback(() => {
    dispatch(prefilledFormDataChanged(null));
    dispatch(updatingStylePresetIdChanged(null));
    dispatch(isModalOpenChanged(false));
  }, [dispatch]);

  useEffect(() => {
    setFormData(null);
  }, []);

  useEffect(() => {
    const convertImageToBlob = async (data: PrefilledFormData | null) => {
      if (!data) {
        setFormData(null);
      } else {
        let file = null;
        if (data.imageUrl) {
          const blob = await convertImageUrlToBlob(data.imageUrl);
          if (blob) {
            file = new File([blob], 'style_preset.png', { type: 'image/png' });
          }
        }
        setFormData({
          ...data,
          image: file,
        });
      }
    };
    convertImageToBlob(prefilledFormData);
  }, [prefilledFormData]);

  return (
    <Modal isOpen={isModalOpen} onClose={handleCloseModal} isCentered size="2xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{modalTitle}</ModalHeader>
        <ModalCloseButton />
        <ModalBody display="flex" flexDir="column" gap={4}>
          {!prefilledFormData || formData ? (
            <StylePresetForm updatingStylePresetId={updatingStylePresetId} formData={formData} />
          ) : (
            <Spinner />
          )}
        </ModalBody>
        <ModalFooter p={2} />
      </ModalContent>
    </Modal>
  );
};
