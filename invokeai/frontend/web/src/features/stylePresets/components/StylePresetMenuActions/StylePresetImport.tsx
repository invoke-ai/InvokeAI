import { Flex, MenuItem, Text } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiSpinner, PiUploadSimpleBold } from 'react-icons/pi';
import { useImportStylePresetsMutation } from 'services/api/endpoints/stylePresets';

export const StylePresetImport = ({ onClose }: { onClose: () => void }) => {
  const [importStylePresets, { isLoading }] = useImportStylePresetsMutation();
  const { t } = useTranslation();

  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (!file) {
        return;
      }
      importStylePresets(file)
        .unwrap()
        .then(() => {
          toast({
            status: 'success',
            title: t('toast.importSuccessful'),
          });
        })
        .catch((error) => {
          toast({
            status: 'error',
            title: t('toast.importFailed'),
            description: error ? `${error.data.detail}` : undefined,
          });
        })
        .finally(() => {
          onClose();
        });
    },
    [importStylePresets, t, onClose]
  );

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  return (
    <MenuItem icon={!isLoading ? <PiUploadSimpleBold /> : <PiSpinner />} alignItems="flex-start" {...getRootProps()}>
      <Flex flexDir="column">
        <Text>{t('stylePresets.importTemplates')}</Text>
        <Text maxW="200px" fontSize="xs" variant="subtext">
          {t('stylePresets.importTemplatesDesc')}
        </Text>
        <Text maxW="200px" fontSize="xs" variant="subtext">
          {t('stylePresets.importTemplatesDesc2')}
        </Text>
      </Flex>
      <input {...getInputProps()} />
    </MenuItem>
  );
};
