import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton, spinAnimation, Text } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import type { ChangeEvent } from 'react';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSpinner, PiUploadBold } from 'react-icons/pi';
import { useImportStylePresetsMutation } from 'services/api/endpoints/stylePresets';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const StylePresetImport = () => {
  const [importStylePresets, { isLoading }] = useImportStylePresetsMutation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { t } = useTranslation();

  const handleClickUpload = useCallback(() => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, [fileInputRef]);

  const handleFileChange = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      if (event.target.files) {
        const file = event.target.files[0];
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
      }
    },
    [importStylePresets, t]
  );

  return (
    <div>
      <input type="file" accept=".csv" onChange={handleFileChange} hidden ref={fileInputRef} />

      <IconButton
        icon={!isLoading ? <PiUploadBold /> : <PiSpinner />}
        tooltip={
          <>
            <Text fontWeight="semibold">{t('stylePresets.importTemplates')}</Text>
            <Text>{t('stylePresets.importTemplatesDesc')}</Text>
          </>
        }
        aria-label={t('stylePresets.importTemplates')}
        onClick={handleClickUpload}
        size="md"
        variant="link"
        w={8}
        h={8}
        sx={isLoading ? loadingStyles : undefined}
        isDisabled={isLoading}
      />
    </div>
  );
};
