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
    let isActive = true;
    const data = stylePresetModalState.prefilledFormData;

    if (!data) {
      void Promise.resolve().then(() => {
        if (isActive) {
          setFormData(null);
        }
      });
      return () => {
        isActive = false;
      };
    }

    if (!data.imageUrl) {
      void Promise.resolve().then(() => {
        if (isActive) {
          setFormData({ ...data, image: null });
        }
      });
      return () => {
        isActive = false;
      };
    }

    void convertImageUrlToBlob(data.imageUrl).then(
      (blob) => {
        if (!isActive) {
          return;
        }
        setFormData({
          ...data,
          image: blob ? new File([blob], 'style_preset.png', { type: 'image/png' }) : null,
        });
      },
      () => {
        if (isActive) {
          setFormData({ ...data, image: null });
        }
      }
    );

    return () => {
      isActive = false;
    };
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
