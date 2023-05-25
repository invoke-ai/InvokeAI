import { useAppDispatch } from 'app/store/storeHooks';
import { isFinite, isString } from 'lodash-es';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import useSetBothPrompts from './usePrompt';
import { allParametersSet, setSeed } from '../store/generationSlice';
import { isImageField } from 'services/types/guards';
import { NUMPY_RAND_MAX } from 'app/constants';
import { initialImageSelected } from '../store/actions';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useAppToaster } from 'app/components/Toaster';
import { ImageDTO } from 'services/api';

export const useParameters = () => {
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const setBothPrompts = useSetBothPrompts();

  /**
   * Sets prompt with toast
   */
  const recallPrompt = useCallback(
    (prompt: unknown) => {
      if (!isString(prompt)) {
        toaster({
          title: t('toast.promptNotSet'),
          description: t('toast.promptNotSetDesc'),
          status: 'warning',
          duration: 2500,
          isClosable: true,
        });
        return;
      }

      setBothPrompts(prompt);
      toaster({
        title: t('toast.promptSet'),
        status: 'info',
        duration: 2500,
        isClosable: true,
      });
    },
    [t, toaster, setBothPrompts]
  );

  /**
   * Sets seed with toast
   */
  const recallSeed = useCallback(
    (seed: unknown) => {
      const s = Number(seed);
      if (!isFinite(s) || (isFinite(s) && !(s >= 0 && s <= NUMPY_RAND_MAX))) {
        toaster({
          title: t('toast.seedNotSet'),
          description: t('toast.seedNotSetDesc'),
          status: 'warning',
          duration: 2500,
          isClosable: true,
        });
        return;
      }

      dispatch(setSeed(s));
      toaster({
        title: t('toast.seedSet'),
        status: 'info',
        duration: 2500,
        isClosable: true,
      });
    },
    [t, toaster, dispatch]
  );

  /**
   * Sets initial image with toast
   */
  const recallInitialImage = useCallback(
    async (image: unknown) => {
      if (!isImageField(image)) {
        toaster({
          title: t('toast.initialImageNotSet'),
          description: t('toast.initialImageNotSetDesc'),
          status: 'warning',
          duration: 2500,
          isClosable: true,
        });
        return;
      }

      dispatch(initialImageSelected(image));
      toaster({
        title: t('toast.initialImageSet'),
        status: 'info',
        duration: 2500,
        isClosable: true,
      });
    },
    [t, toaster, dispatch]
  );

  /**
   * Sets image as initial image with toast
   */
  const sendToImageToImage = useCallback(
    (image: ImageDTO) => {
      dispatch(initialImageSelected(image));
    },
    [dispatch]
  );

  const recallAllParameters = useCallback(
    (image: ImageDTO | undefined) => {
      const type = image?.metadata?.type;
      if (['txt2img', 'img2img', 'inpaint'].includes(String(type))) {
        dispatch(allParametersSet(image));

        if (image?.metadata?.type === 'img2img') {
          dispatch(setActiveTab('img2img'));
        } else if (image?.metadata?.type === 'txt2img') {
          dispatch(setActiveTab('txt2img'));
        }

        toaster({
          title: t('toast.parametersSet'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toaster({
          title: t('toast.parametersNotSet'),
          description: t('toast.parametersNotSetDesc'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [t, toaster, dispatch]
  );

  return {
    recallPrompt,
    recallSeed,
    recallInitialImage,
    sendToImageToImage,
    recallAllParameters,
  };
};
