import { Box, IconButton, Image } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import type { AnyModelConfig } from 'services/api/types';

import { Button } from '@invoke-ai/ui-library';
import { useDropzone } from 'react-dropzone';
import { PiArrowCounterClockwiseBold, PiUploadSimpleBold } from 'react-icons/pi';
import IAIDndImageIcon from 'common/components/IAIDndImageIcon';

const ModelImageUpload = (props: UseControllerProps<AnyModelConfig>) => {
  const { field } = useController(props);

  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];

      if (!file) {
        return;
      }

      field.onChange(file);
    },
    [field]
  );

  const handleResetControlImage = useCallback(() => {
    field.onChange(undefined);
  }, [field]);

  console.log('field', field);

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  if (field.value) {
    return (
      <Box>
        <Image
          src={URL.createObjectURL(field.value)}
          objectFit="contain"
          maxW="full"
          maxH="200px"
          borderRadius="base"
        />
        <IconButton
          onClick={handleResetControlImage}
          aria-label="reset this image"
          tooltip="reset this image"
          icon={<PiArrowCounterClockwiseBold size={16} />}
          size="sm"
          variant="link"
        //   sx={sx}
        />
      </Box>
    );
  }

  return (
    <>
      <Button leftIcon={<PiUploadSimpleBold />} {...getRootProps()} pointerEvents="auto">
        Upload Image
      </Button>
      <input {...getInputProps()} />
    </>
  );
};

export default typedMemo(ModelImageUpload);

