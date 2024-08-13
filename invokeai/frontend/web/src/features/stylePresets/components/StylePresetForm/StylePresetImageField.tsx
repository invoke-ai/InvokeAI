import { Box, Button, Flex, Icon, IconButton, Image, Tooltip } from '@invoke-ai/ui-library';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiUploadSimpleBold } from 'react-icons/pi';

import type { StylePresetFormData } from './StylePresetForm';

export const StylePresetImageField = (props: UseControllerProps<StylePresetFormData, 'image'>) => {
  const { field } = useController(props);
  const { t } = useTranslation();
  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (file) {
        field.onChange(file);
      }
    },
    [field]
  );

  const handleResetImage = useCallback(() => {
    field.onChange(null);
  }, [field]);

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  if (field.value) {
    return (
      <Box position="relative" flexShrink={0}>
        <Image
          src={URL.createObjectURL(field.value)}
          objectFit="cover"
          objectPosition="50% 50%"
          w={65}
          h={65}
          minWidth={65}
          borderRadius="base"
        />
        <IconButton
          position="absolute"
          insetInlineEnd={0}
          insetBlockStart={0}
          onClick={handleResetImage}
          aria-label={t('stylePresets.deleteImage')}
          tooltip={t('stylePresets.deleteImage')}
          icon={<PiArrowCounterClockwiseBold />}
          size="md"
          variant="ghost"
        />
      </Box>
    );
  }

  return (
    <>
      <Tooltip label={t('stylePresets.uploadImage')}>
        <Flex
          as={Button}
          w={65}
          h={65}
          opacity={0.3}
          borderRadius="base"
          alignItems="center"
          justifyContent="center"
          flexShrink={0}
          {...getRootProps()}
        >
          <Icon as={PiUploadSimpleBold} w={8} h={8} />
        </Flex>
      </Tooltip>
      <input {...getInputProps()} />
    </>
  );
};
