import { Box, IconButton, Image } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback, useState } from 'react';
import type { Control } from 'react-hook-form';
import { useController, useWatch } from 'react-hook-form';

import { Button } from '@invoke-ai/ui-library';
import { useDropzone } from 'react-dropzone';
import { PiArrowCounterClockwiseBold, PiUploadSimpleBold } from 'react-icons/pi';
import { UpdateModelArg, buildModelsUrl } from 'services/api/endpoints/models';
import { useTranslation } from 'react-i18next';

type Props = {
  control: Control<UpdateModelArg['body']>;
};

const ModelImageUpload = ({ control }: Props) => {
  const key = useWatch({ control, name: 'key' });
  const [image, setImage] = useState<string | undefined>(buildModelsUrl(`i/${key}/image`));
  const { field } = useController({ control, name: 'image' });
  const { t } = useTranslation();

  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];

      if (!file) {
        return;
      }

      field.onChange(file);
      setImage(URL.createObjectURL(file));
    },
    [field]
  );

  const handleResetImage = useCallback(() => {
    field.onChange(undefined);
    setImage(undefined);
  }, [field]);

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  if (image) {
    return (
      <Box
        position="relative"
        _hover={{ filter: 'brightness(50%)' }}
        transition="filter ease 0.2s"
      >
        <Image
          onError={() => setImage(undefined)}
          src={image}
          objectFit="contain"
          maxW="full"
          maxH="100px"
          borderRadius="base"
        />
        <IconButton
          position="absolute"
          top="1"
          right="1"
          onClick={handleResetImage}
          aria-label={t('modelManager.resetImage')}
          tooltip={t('modelManager.resetImage')}
          icon={<PiArrowCounterClockwiseBold size={16} />}
          size="sm"
          variant="link"
        />
      </Box>
    );
  }

  return (
    <>
      <Button leftIcon={<PiUploadSimpleBold />} {...getRootProps()} pointerEvents="auto">
        {t('modelManager.uploadImage')}
      </Button>
      <input {...getInputProps()} />
    </>
  );
};

export default typedMemo(ModelImageUpload);
