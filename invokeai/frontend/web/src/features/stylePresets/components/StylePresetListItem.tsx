import { Badge, ConfirmationAlertDialog, Flex, IconButton, Text, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useImageUrlToBlob } from 'common/hooks/useImageUrlToBlob';
import {
  isModalOpenChanged,
  prefilledFormDataChanged,
  updatingStylePresetIdChanged,
} from 'features/stylePresets/store/stylePresetModalSlice';
import { activeStylePresetChanged, isMenuOpenChanged } from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPencilBold, PiTrashBold } from 'react-icons/pi';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';
import { useDeleteStylePresetMutation } from 'services/api/endpoints/stylePresets';

import StylePresetImage from './StylePresetImage';

export const StylePresetListItem = ({ preset }: { preset: StylePresetRecordWithImage }) => {
  const dispatch = useAppDispatch();
  const [deleteStylePreset] = useDeleteStylePresetMutation();
  const activeStylePreset = useAppSelector((s) => s.stylePreset.activeStylePreset);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const imageUrlToBlob = useImageUrlToBlob();
  const { t } = useTranslation();

  const handleClickEdit = useCallback(
    async (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      const { name, preset_data } = preset;
      const { positive_prompt, negative_prompt } = preset_data;
      let imageBlob = null;
      if (preset.image) {
        imageBlob = await imageUrlToBlob(preset.image, 100);
      }

      dispatch(
        prefilledFormDataChanged({
          name,
          positivePrompt: positive_prompt,
          negativePrompt: negative_prompt,
          image: imageBlob ? new File([imageBlob], `style_preset_${preset.id}.png`, { type: 'image/png' }) : null,
        })
      );

      dispatch(updatingStylePresetIdChanged(preset.id));
      dispatch(isModalOpenChanged(true));
    },
    [dispatch, preset, imageUrlToBlob]
  );

  const handleClickApply = useCallback(async () => {
    dispatch(activeStylePresetChanged(preset));
    dispatch(isMenuOpenChanged(false));
  }, [dispatch, preset]);

  const handleClickDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      onOpen();
    },
    [onOpen]
  );

  const handleDeletePreset = useCallback(async () => {
    try {
      await deleteStylePreset(preset.id);
      toast({
        status: 'success',
        title: t('stylePresets.templateDeleted'),
      });
    } catch (error) {
      toast({
        status: 'error',
        title: t('stylePresets.unableToDeleteTemplate'),
      });
    }
  }, [preset, t, deleteStylePreset]);

  return (
    <>
      <Flex
        gap="4"
        onClick={handleClickApply}
        cursor="pointer"
        _hover={{ backgroundColor: 'base.750' }}
        padding="10px"
        borderRadius="base"
        alignItems="flex-start"
        w="full"
      >
        <StylePresetImage presetImageUrl={preset.image} />
        <Flex flexDir="column" w="full">
          <Flex w="full" justifyContent="space-between" alignItems="flex-start">
            <Flex alignItems="center" gap="2">
              <Text fontSize="md" noOfLines={2}>
                {preset.name}
              </Text>
              {activeStylePreset && activeStylePreset.id === preset.id && (
                <Badge
                  color="invokeBlue.400"
                  borderColor="invokeBlue.700"
                  borderWidth={1}
                  bg="transparent"
                  flexShrink={0}
                >
                  {t('stylePresets.active')}
                </Badge>
              )}
            </Flex>

            {!preset.is_default && (
              <Flex alignItems="center" gap="1">
                <IconButton
                  size="sm"
                  variant="ghost"
                  aria-label={t('stylePresets.editTemplate')}
                  onClick={handleClickEdit}
                  icon={<PiPencilBold />}
                />
                <IconButton
                  size="sm"
                  variant="ghost"
                  aria-label={t('stylePresets.deleteTemplate')}
                  onClick={handleClickDelete}
                  colorScheme="error"
                  icon={<PiTrashBold />}
                />
              </Flex>
            )}
          </Flex>

          <Flex flexDir="column" gap="1">
            <Text fontSize="xs">
              <Text as="span" fontWeight="semibold">
                {t('stylePresets.positivePrompt')}:
              </Text>{' '}
              {preset.preset_data.positive_prompt}
            </Text>
            <Text fontSize="xs">
              <Text as="span" fontWeight="semibold">
                {t('stylePresets.negativePrompt')}:
              </Text>{' '}
              {preset.preset_data.negative_prompt}
            </Text>
          </Flex>
        </Flex>
      </Flex>
      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('stylePresets.deleteTemplate')}
        acceptCallback={handleDeletePreset}
        acceptButtonText="Delete"
      >
        <p>{t('stylePresets.deleteTemplate2')}</p>
      </ConfirmationAlertDialog>
    </>
  );
};
