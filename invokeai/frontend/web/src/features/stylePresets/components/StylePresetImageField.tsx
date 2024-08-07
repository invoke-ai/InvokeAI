import { Tooltip, Flex, Button, Icon, Box, Image, IconButton } from '@invoke-ai/ui-library';
import { t } from 'i18next';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { PiArrowCounterClockwiseBold, PiUploadSimpleBold } from 'react-icons/pi';
import { useController, UseControllerProps } from 'react-hook-form';
import { StylePresetFormData } from './StylePresetForm';

export const StylePresetImageField = (props: UseControllerProps<StylePresetFormData>) => {
  const { field } = useController(props);
  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (file) {
        field.onChange(file);
      }
    },
    [field, t]
  );

  const handleResetImage = useCallback(() => {
    field.onChange(null);
  }, []);

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
          src={URL.createObjectURL(field.value as File)}
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
          aria-label={t('modelManager.deleteModelImage')}
          tooltip={t('modelManager.deleteModelImage')}
          icon={<PiArrowCounterClockwiseBold />}
          size="md"
          variant="ghost"
        />
      </Box>
    );
  }

  return (
    <>
      <Tooltip label={t('modelManager.uploadImage')}>
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
