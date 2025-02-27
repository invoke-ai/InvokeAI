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
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { convertImageUrlToBlob } from 'common/util/convertImageUrlToBlob';
import type { PrefilledFormData } from 'features/stylePresets/store/stylePresetModal';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { StylePresetFormData } from './StylePresetForm';
import { StylePresetForm } from './StylePresetForm';

export const StylePresetModal = () => {
  useAssertSingleton('StylePresetModal');
  const [formData, setFormData] = useState<StylePresetFormData | null>(null);
  const { t } = useTranslation();
  const stylePresetModalState = useStore($stylePresetModalState);

  const modalTitle = useMemo(() => {
    return stylePresetModalState.updatingStylePresetId
      ? t('stylePresets.updatePromptTemplate')
      : t('stylePresets.createPromptTemplate');
  }, [stylePresetModalState.updatingStylePresetId, t]);

  const handleCloseModal = useCallback(() => {
    $stylePresetModalState.set({
      prefilledFormData: null,
      updatingStylePresetId: null,
      isModalOpen: false,
    });
  }, []);

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
          try {
            const blob = await convertImageUrlToBlob(data.imageUrl);
            if (blob) {
              file = new File([blob], 'style_preset.png', { type: 'image/png' });
            }
          } catch (error) {
            // do nothing
          }
        }
        setFormData({
          ...data,
          image: file,
        });
      }
    };
    convertImageToBlob(stylePresetModalState.prefilledFormData);
  }, [stylePresetModalState.prefilledFormData]);

  return (
    <Modal isOpen={stylePresetModalState.isModalOpen} onClose={handleCloseModal} isCentered size="2xl" useInert={false}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{modalTitle}</ModalHeader>
        <ModalCloseButton />
        <ModalBody display="flex" flexDir="column" gap={4}>
          {!stylePresetModalState.prefilledFormData || formData ? (
            <StylePresetForm updatingStylePresetId={stylePresetModalState.updatingStylePresetId} formData={formData} />
          ) : (
            <Spinner />
          )}
        </ModalBody>
        <ModalFooter p={2} />
      </ModalContent>
    </Modal>
  );
};
