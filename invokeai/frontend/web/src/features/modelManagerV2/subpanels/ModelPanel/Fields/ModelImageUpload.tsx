import { Box, Button, Flex, Icon, IconButton, Image, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { typedMemo } from 'common/util/typedMemo';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiUploadSimpleBold } from 'react-icons/pi';
import { useDeleteModelImageMutation, useUpdateModelImageMutation } from 'services/api/endpoints/models';

type Props = {
  model_key: string | null;
  model_image?: string | null;
};

const ModelImageUpload = ({ model_key, model_image }: Props) => {
  const dispatch = useAppDispatch();
  const [image, setImage] = useState<string | null>(model_image || null);
  const { t } = useTranslation();

  const [updateModelImage] = useUpdateModelImageMutation();
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
    [dispatch, model_key, t, updateModelImage]
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
  }, [dispatch, model_key, t, deleteModelImage]);

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
          height="100px"
          width="100px"
          minWidth="100px"
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
          w={100}
          h={100}
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
