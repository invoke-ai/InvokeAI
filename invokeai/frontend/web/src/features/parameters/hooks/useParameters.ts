import { UseToastOptions, useToast } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { isFinite, isString } from 'lodash-es';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import useSetBothPrompts from './usePrompt';
import { initialImageSelected, setSeed } from '../store/generationSlice';
import { isImage, isImageField } from 'services/types/guards';
import { NUMPY_RAND_MAX } from 'app/constants';

export const useParameters = () => {
  const dispatch = useAppDispatch();
  const toast = useToast();
  const { t } = useTranslation();
  const setBothPrompts = useSetBothPrompts();

  /**
   * Sets prompt with toast
   */
  const recallPrompt = useCallback(
    (prompt: unknown) => {
      if (!isString(prompt)) {
        toast({
          title: t('toast.promptNotSet'),
          description: t('toast.promptNotSetDesc'),
          status: 'warning',
          duration: 2500,
          isClosable: true,
        });
        return;
      }

      setBothPrompts(prompt);
      toast({
        title: t('toast.promptSet'),
        status: 'info',
        duration: 2500,
        isClosable: true,
      });
    },
    [t, toast, setBothPrompts]
  );

  /**
   * Sets seed with toast
   */
  const recallSeed = useCallback(
    (seed: unknown) => {
      const s = Number(seed);
      if (!isFinite(s) || (isFinite(s) && !(s >= 0 && s <= NUMPY_RAND_MAX))) {
        toast({
          title: t('toast.seedNotSet'),
          description: t('toast.seedNotSetDesc'),
          status: 'warning',
          duration: 2500,
          isClosable: true,
        });
        return;
      }

      dispatch(setSeed(s));
      toast({
        title: t('toast.seedSet'),
        status: 'info',
        duration: 2500,
        isClosable: true,
      });
    },
    [t, toast, dispatch]
  );

  /**
   * Sets initial image with toast
   */
  const recallInitialImage = useCallback(
    async (image: unknown) => {
      if (!isImageField(image)) {
        toast({
          title: t('toast.initialImageNotSet'),
          description: t('toast.initialImageNotSetDesc'),
          status: 'warning',
          duration: 2500,
          isClosable: true,
        });
        return;
      }

      dispatch(
        initialImageSelected({ name: image.image_name, type: image.image_type })
      );
      toast({
        title: t('toast.initialImageSet'),
        status: 'info',
        duration: 2500,
        isClosable: true,
      });
    },
    [t, toast, dispatch]
  );

  /**
   * Sets image as initial image with toast
   */
  const sendToImageToImage = useCallback(
    (image: unknown) => {
      if (!isImage(image)) {
        toast({
          title: t('toast.imageNotLoaded'),
          description: t('toast.imageNotLoadedDesc'),
          status: 'warning',
          duration: 2500,
          isClosable: true,
        });
        return;
      }

      dispatch(initialImageSelected({ name: image.name, type: image.type }));
      toast({
        title: t('toast.sentToImageToImage'),
        status: 'info',
        duration: 2500,
        isClosable: true,
      });
    },
    [t, toast, dispatch]
  );

  return { recallPrompt, recallSeed, recallInitialImage, sendToImageToImage };
};
