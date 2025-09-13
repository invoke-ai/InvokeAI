import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, IconButton, Image } from '@invoke-ai/ui-library';
import { dropzoneAccept } from 'common/hooks/useImageUploadButton';
import { typedMemo } from 'common/util/typedMemo';
import { toast } from 'features/toast/toast';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiUploadBold } from 'react-icons/pi';
import { useDeleteModelImageMutation, useUpdateModelImageMutation } from 'services/api/endpoints/models';

const sharedSx: SystemStyleObject = {
  w: 108,
  h: 108,
  fontSize: 36,
  borderRadius: 'base',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  bg: 'base.800',
  borderWidth: '1px',
  borderStyle: 'solid',
  borderColor: 'base.700',
  flexShrink: 0,
};

type Props = {
  model_key: string | null;
  model_image?: string | null;
};

const ModelImageUpload = ({ model_key, model_image }: Props) => {
  const [image, setImage] = useState<string | null>(model_image || null);
  const { t } = useTranslation();

  const [updateModelImage, request] = useUpdateModelImageMutation();
  const [deleteModelImage] = useDeleteModelImageMutation();

  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];

      if (!file || !model_key) {
        return;
      }

      updateModelImage({ key: model_key, image: file })
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
    [model_key, t, updateModelImage]
  );

  const handleResetImage = useCallback(() => {
    if (!model_key) {
      return;
    }
    setImage(null);
    deleteModelImage(model_key)
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
  }, [model_key, t, deleteModelImage]);

  const { getInputProps, getRootProps } = useDropzone({
    accept: dropzoneAccept,
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
          minWidth={108}
          borderRadius="base"
          sx={sharedSx}
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
      <IconButton
        variant="ghost"
        aria-label={t('modelManager.uploadImage')}
        tooltip={t('modelManager.uploadImage')}
        fontSize={36}
        icon={<PiUploadBold />}
        sx={sharedSx}
        isLoading={request.isLoading}
        {...getRootProps()}
      />
      <input {...getInputProps()} />
    </>
  );
};

export default typedMemo(ModelImageUpload);
