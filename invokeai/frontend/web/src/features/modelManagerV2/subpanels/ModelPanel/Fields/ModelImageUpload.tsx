import { Box, Button, Flex, Icon, IconButton, Image, Tooltip } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { toast } from 'features/toast/toast';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiUploadSimpleBold } from 'react-icons/pi';
import { useDeleteModelImageMutation, useUpdateModelImageMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

type Props = {
  modelConfig: AnyModelConfig;
};

const ModelImageUpload = ({ modelConfig }: Props) => {
  const { key, cover_image } = modelConfig;
  const [image, setImage] = useState<string | null>(cover_image || null);
  const { t } = useTranslation();

  const [updateModelImage] = useUpdateModelImageMutation();
  const [deleteModelImage] = useDeleteModelImageMutation();

  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];

      if (!file) {
        return;
      }

      updateModelImage({ key, image: file })
        .unwrap()
        .then(() => {
          setImage(URL.createObjectURL(file));
          toast({
            id: 'MODEL_IMAGE_UPDATED',
            title: t('modelManager.modelImageUpdated'),
            status: 'success',
          });
        })
        .catch(() => {
          toast({
            id: 'MODEL_IMAGE_UPDATE_FAILED',
            title: t('modelManager.modelImageUpdateFailed'),
            status: 'error',
          });
        });
    },
    [key, t, updateModelImage]
  );

  const handleResetImage = useCallback(() => {
    setImage(null);
    deleteModelImage(key)
      .unwrap()
      .then(() => {
        toast({
          id: 'MODEL_IMAGE_DELETED',
          title: t('modelManager.modelImageDeleted'),
          status: 'success',
        });
      })
      .catch(() => {
        toast({
          id: 'MODEL_IMAGE_DELETE_FAILED',
          title: t('modelManager.modelImageDeleteFailed'),
          status: 'error',
        });
      });
  }, [deleteModelImage, key, t]);

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  if (image) {
    return (
      <Box position="relative" flexShrink={0}>
        <Image
          src={image}
          objectFit="cover"
          objectPosition="50% 50%"
          height={108}
          width={108}
          minWidth={108}
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
          w={108}
          h={108}
          opacity={0.3}
          borderRadius="base"
          alignItems="center"
          justifyContent="center"
          flexShrink={0}
          {...getRootProps()}
        >
          <Icon as={PiUploadSimpleBold} w={16} h={16} />
        </Flex>
      </Tooltip>
      <input {...getInputProps()} />
    </>
  );
};

export default typedMemo(ModelImageUpload);
