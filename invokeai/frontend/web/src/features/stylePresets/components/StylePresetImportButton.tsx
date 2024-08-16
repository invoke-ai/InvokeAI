import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton, ListItem, spinAnimation, Text, UnorderedList } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiSpinner, PiUploadSimpleBold } from 'react-icons/pi';
import { useImportStylePresetsMutation } from 'services/api/endpoints/stylePresets';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const StylePresetImportButton = () => {
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
        });
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
        icon={!isLoading ? <PiUploadSimpleBold /> : <PiSpinner />}
        tooltip={<TooltipContent />}
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

const TooltipContent = () => {
  const { t } = useTranslation();
  return (
    <Flex flexDir="column">
      <Text pb={1} fontWeight="semibold">
        {t('stylePresets.importTemplates')}
      </Text>
      <Text>{t('stylePresets.acceptedColumnsKeys')}</Text>
      <UnorderedList>
        <ListItem>{t('stylePresets.nameColumn')}</ListItem>
        <ListItem>{t('stylePresets.positivePromptColumn')}</ListItem>
        <ListItem>{t('stylePresets.negativePromptColumn')}</ListItem>
      </UnorderedList>
    </Flex>
  );
};
