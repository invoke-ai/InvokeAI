import { IconButton, Menu, MenuButton, MenuItem, MenuList, Text } from '@invoke-ai/ui-library';
import {
  promptExpansionRequested,
  promptGenerationFromUploadRequested,
} from 'app/store/middleware/listenerMiddleware/listeners/addPromptExpansionRequestedListener';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagicWandBold } from 'react-icons/pi';

export const PromptExpansionMenu = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onClickExpandPrompt = useCallback(() => {
    dispatch(promptExpansionRequested());
  }, [dispatch]);

  const onClickUploadImage = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const onFileSelected = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        dispatch(promptGenerationFromUploadRequested({ file }));
        // Reset the input so the same file can be selected again
        event.target.value = '';
      }
    },
    [dispatch]
  );

  return (
    <>
      <Menu>
        <MenuButton
          as={IconButton}
          icon={<PiMagicWandBold size={16} />}
          size="sm"
          borderRadius="100%"
          colorScheme="invokeYellow"
        />
        <MenuList>
          <MenuItem onClick={onClickExpandPrompt}>
            <Text>{t('prompt.expandCurrentPrompt')}</Text>
          </MenuItem>
          <MenuItem onClick={onClickUploadImage}>
            <Text>{t('prompt.uploadImageForPromptGeneration')}</Text>
          </MenuItem>
        </MenuList>
      </Menu>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/webp"
        onChange={onFileSelected}
        style={{ display: 'none' }}
      />
    </>
  );
};
