import { IconButton } from '@invoke-ai/ui-library';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const StylePresetCreateButton = () => {
  const handleClickAddNew = useCallback(() => {
    $stylePresetModalState.set({
      prefilledFormData: null,
      updatingStylePresetId: null,
      isModalOpen: true,
    });
  }, []);

  const { t } = useTranslation();

  return (
    <IconButton
      icon={<PiPlusBold />}
      tooltip={t('stylePresets.createPromptTemplate')}
      aria-label={t('stylePresets.createPromptTemplate')}
      onClick={handleClickAddNew}
      size="md"
      variant="ghost"
      w={8}
      h={8}
    />
  );
};
