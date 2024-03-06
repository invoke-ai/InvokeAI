import { Box, IconButton, Image } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback, useState } from 'react';
import { useAppDispatch } from 'app/store/storeHooks';

import { Button } from '@invoke-ai/ui-library';
import { useDropzone } from 'react-dropzone';
import { PiArrowCounterClockwiseBold, PiUploadSimpleBold } from 'react-icons/pi';
import { useUpdateModelImageMutation, useDeleteModelImageMutation } from 'services/api/endpoints/models';
import { useTranslation } from 'react-i18next';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';

type Props = {
    model_key: string | null;
    model_image: string | null;
  };
  
  const ModelImageUpload = ({ model_key, model_image }: Props) => {
  const dispatch = useAppDispatch();
  const [image, setImage] = useState<string | null>(model_image);
  const { t } = useTranslation();

  const [updateModelImage] = useUpdateModelImageMutation();
  const [deleteModelImage] = useDeleteModelImageMutation();

  const onDropAccepted = useCallback(
    (files: File[]) => {
      const file = files[0];

      if (!file) {
        return;
      }

      setImage(URL.createObjectURL(file));

      updateModelImage({ key: model_key, image: file })
      .unwrap()
      .then(() => {
        dispatch(
          addToast(
            makeToast({
              title: t('modelManager.modelImageUpdated'),
              status: 'success',
            })
          )
        );
      })
      .catch((_) => {
        dispatch(
          addToast(
            makeToast({
              title: t('modelManager.modelImageUpdateFailed'),
              status: 'error',
            })
          )
        );
      });
    },
    []
  );

  const handleResetImage = useCallback(() => {
    if (!model_key) {
      return;
    }
    setImage(null);
    deleteModelImage(model_key)
    .unwrap()
    .then(() => {
      dispatch(
        addToast(
          makeToast({
            title: t('modelManager.modelImageDeleted'),
            status: 'success',
          })
        )
      );
    })
    .catch((_) => {
      dispatch(
        addToast(
          makeToast({
            title: t('modelManager.modelImageDeleteFailed'),
            status: 'error',
          })
        )
      );
    });
  }, []);

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
      >
        <Image
          src={image}
          objectFit="cover"
          objectPosition="50% 50%"
          height="100px"
          width="100px"
          borderRadius="base"
        />
        <IconButton
          position="absolute"
          top="1"
          right="1"
          onClick={handleResetImage}
          aria-label={t('modelManager.deleteModelImage')}
          tooltip={t('modelManager.deleteModelImage')}
          icon={<PiArrowCounterClockwiseBold size={16} />}
          size="sm"
          variant="link"
          _hover={{ color: 'base.100' }}
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
