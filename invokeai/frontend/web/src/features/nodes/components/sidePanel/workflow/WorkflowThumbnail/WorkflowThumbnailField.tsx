import { Box, Button, Flex, Icon, IconButton, Image, Tooltip } from '@invoke-ai/ui-library';
import { convertImageUrlToBlob } from 'common/util/convertImageUrlToBlob';
import { useCallback, useEffect, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiUploadSimpleBold } from 'react-icons/pi';

export const WorkflowThumbnailField = ({
  imageUrl,
  onChange,
}: {
  imageUrl?: string | null;
  onChange: (localThumbnailUrl: string | null) => void;
}) => {
  const [thumbnail, setThumbnail] = useState<File | null>(null);

  const syncThumbnail = useCallback(async (imageUrl?: string | null) => {
    if (!imageUrl) {
      setThumbnail(null);
      return;
    }
    try {
      const blob = await convertImageUrlToBlob(imageUrl);
      if (blob) {
        const file = new File([blob], 'workflow.png', { type: 'image/png' });
        setThumbnail(file);
      }
    } catch (error) {
      setThumbnail(null);
    }
  }, []);

  useEffect(() => {
    syncThumbnail(imageUrl);
  }, [imageUrl, syncThumbnail]);

  const { t } = useTranslation();

  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (file) {
        setThumbnail(file);
        onChange(URL.createObjectURL(file));
      }
    },
    [onChange]
  );

  const handleResetImage = useCallback(() => {
    setThumbnail(null);
    onChange(null);
  }, [onChange]);

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  if (thumbnail) {
    return (
      <Box position="relative" flexShrink={0}>
        <Image
          src={URL.createObjectURL(thumbnail)}
          objectFit="cover"
          objectPosition="50% 50%"
          w={100}
          h={100}
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
          w={100}
          h={100}
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
