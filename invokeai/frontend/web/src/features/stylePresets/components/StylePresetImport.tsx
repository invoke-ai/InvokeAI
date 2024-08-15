import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton, spinAnimation, Text } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiSpinner, PiUploadBold } from 'react-icons/pi';
import { useImportStylePresetsMutation } from 'services/api/endpoints/stylePresets';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const StylePresetImport = () => {
  const [importStylePresets, { isLoading }] = useImportStylePresetsMutation();
  const { t } = useTranslation();

  const onDropAccepted = useCallback(
    async (files: File[]) => {
      const file = files[0];
      if (!file) {
        return;
      }
      try {
        await importStylePresets(file).unwrap();
        toast({
          status: 'success',
          title: t('toast.importSuccessful'),
        });
      } catch (error) {
        toast({
          status: 'error',
          title: t('toast.importFailed'),
        });
      }
    },
    [importStylePresets, t]
  );

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  return (
    <>
      <IconButton
        icon={!isLoading ? <PiUploadBold /> : <PiSpinner />}
        tooltip={
          <>
            <Text fontWeight="semibold">{t('stylePresets.importTemplates')}</Text>
            <Text>{t('stylePresets.importTemplatesDesc')}</Text>
          </>
        }
        aria-label={t('stylePresets.importTemplates')}
        size="md"
        variant="link"
        w={8}
        h={8}
        sx={isLoading ? loadingStyles : undefined}
        isDisabled={isLoading}
        {...getRootProps()}
      />
      <input {...getInputProps()} />
    </>
  );
};
