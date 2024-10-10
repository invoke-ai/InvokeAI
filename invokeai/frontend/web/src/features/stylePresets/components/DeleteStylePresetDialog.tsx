import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { toast } from 'features/toast/toast';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';
import { useDeleteStylePresetMutation } from 'services/api/endpoints/stylePresets';

const $presetToDelete = atom<StylePresetRecordWithImage | null>(null);
const clearPresetToDelete = () => $presetToDelete.set(null);

export const useDeleteStylePreset = () => {
  const deletePreset = useCallback((preset: StylePresetRecordWithImage) => {
    $presetToDelete.set(preset);
  }, []);

  return deletePreset;
};

export const DeleteStylePresetDialog = memo(() => {
  useAssertSingleton('DeleteStylePresetDialog');
  const { t } = useTranslation();
  const presetToDelete = useStore($presetToDelete);
  const [_deleteStylePreset] = useDeleteStylePresetMutation();

  const deleteStylePreset = useCallback(async () => {
    if (!presetToDelete) {
      return;
    }
    try {
      await _deleteStylePreset(presetToDelete.id).unwrap();
      toast({
        status: 'success',
        title: t('stylePresets.templateDeleted'),
      });
    } catch (error) {
      toast({
        status: 'error',
        title: t('stylePresets.unableToDeleteTemplate'),
      });
    }
  }, [presetToDelete, _deleteStylePreset, t]);

  return (
    <ConfirmationAlertDialog
      isOpen={presetToDelete !== null}
      onClose={clearPresetToDelete}
      title={t('stylePresets.deleteTemplate')}
      acceptCallback={deleteStylePreset}
      acceptButtonText={t('common.delete')}
      cancelButtonText={t('common.cancel')}
      useInert={false}
    >
      <Text>{t('stylePresets.deleteTemplate2')}</Text>
    </ConfirmationAlertDialog>
  );
});

DeleteStylePresetDialog.displayName = 'DeleteStylePresetDialog';
