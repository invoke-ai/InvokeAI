import { IconButton, Menu, MenuButton, MenuItem, MenuList, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppStore } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { WrappedError } from 'common/util/result';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagicWandBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import { expandPrompt } from './expand';
import { promptExpansionApi } from './state';

export const PromptExpansionMenu = () => {
  const { dispatch, getState } = useAppStore();
  const { t } = useTranslation();
  const { isPending } = useStore(promptExpansionApi.$state);

  const onUploadStarted = useCallback(() => {
    promptExpansionApi.setPending();
  }, []);

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      promptExpansionApi.setPending(imageDTO);
      expandPrompt({ dispatch, getState, imageDTO });
    },
    [dispatch, getState]
  );

  const onUploadError = useCallback(
    (error: unknown) => {
      const wrappedError = WrappedError.wrap(error);
      promptExpansionApi.setError(wrappedError);
      toast({
        id: 'UPLOAD_AND_PROMPT_GENERATION_FAILED',
        title: t('toast.uploadAndPromptGenerationFailed'),
        status: 'error',
      });
    },
    [t]
  );

  const uploadApi = useImageUploadButton({
    allowMultiple: false,
    onUpload,
    onUploadStarted,
    onError: onUploadError,
  });

  const onClickExpandPrompt = useCallback(() => {
    promptExpansionApi.setPending();
    expandPrompt({ dispatch, getState });
  }, [dispatch, getState]);

  return (
    <>
      <Menu>
        <MenuButton
          as={IconButton}
          icon={<PiMagicWandBold size={16} />}
          size="sm"
          borderRadius="100%"
          colorScheme="invokeYellow"
          isDisabled={isPending}
        />
        <MenuList>
          <MenuItem onClick={onClickExpandPrompt} isDisabled={isPending}>
            <Text>{t('prompt.expandCurrentPrompt')}</Text>
          </MenuItem>
          <MenuItem {...uploadApi.getUploadButtonProps()} isDisabled={isPending}>
            <Text>{t('prompt.uploadImageForPromptGeneration')}</Text>
          </MenuItem>
        </MenuList>
      </Menu>
      <input {...uploadApi.getUploadInputProps()} />
    </>
  );
};
