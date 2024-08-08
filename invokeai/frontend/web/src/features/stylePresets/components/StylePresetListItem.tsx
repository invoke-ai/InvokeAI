import { Badge, ConfirmationAlertDialog, Flex, IconButton, Text, useDisclosure } from '@invoke-ai/ui-library';
import type { MouseEvent } from 'react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isModalOpenChanged,
  prefilledFormDataChanged,
  updatingStylePresetIdChanged,
} from 'features/stylePresets/store/stylePresetModalSlice';
import { activeStylePresetChanged, isMenuOpenChanged } from 'features/stylePresets/store/stylePresetSlice';
import { useCallback } from 'react';
import { PiPencilBold, PiTrashBold } from 'react-icons/pi';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';
import { useDeleteStylePresetMutation } from 'services/api/endpoints/stylePresets';
import StylePresetImage from './StylePresetImage';
import { useImageUrlToBlob } from 'common/hooks/useImageUrlToBlob';

export const StylePresetListItem = ({ preset }: { preset: StylePresetRecordWithImage }) => {
  const dispatch = useAppDispatch();
  const [deleteStylePreset] = useDeleteStylePresetMutation();
  const activeStylePreset = useAppSelector((s) => s.stylePreset.activeStylePreset);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const imageUrlToBlob = useImageUrlToBlob();

  const handleClickEdit = useCallback(
    async (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      const { name, preset_data } = preset;
      const { positive_prompt, negative_prompt } = preset_data;
      let imageBlob = null;
      if (preset.image) {
        imageBlob = await imageUrlToBlob(preset.image);
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
    [dispatch, preset]
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
    [dispatch, preset]
  );

  const handleDeletePreset = useCallback(async () => {
    try {
      await deleteStylePreset(preset.id);
    } catch (error) {}
  }, [preset]);

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
                  Active
                </Badge>
              )}
            </Flex>

            <Flex alignItems="center" gap="1">
              <IconButton
                size="sm"
                variant="ghost"
                aria-label="Edit"
                onClick={handleClickEdit}
                icon={<PiPencilBold />}
              />
              <IconButton
                size="sm"
                variant="ghost"
                aria-label="Delete"
                onClick={handleClickDelete}
                colorScheme="error"
                icon={<PiTrashBold />}
              />
            </Flex>
          </Flex>

          <Flex flexDir="column">
            <Text fontSize="xs">
              <Text as="span" fontWeight="semibold">
                Positive prompt:
              </Text>{' '}
              {preset.preset_data.positive_prompt}
            </Text>
            <Text fontSize="xs">
              <Text as="span" fontWeight="semibold">
                Negative prompt:
              </Text>{' '}
              {preset.preset_data.negative_prompt}
            </Text>
          </Flex>
        </Flex>
      </Flex>
      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={'Delete preset'}
        acceptCallback={handleDeletePreset}
        acceptButtonText={'Delete'}
      >
        <p>{'Delete Preset?'}</p>
        <br />
      </ConfirmationAlertDialog>
    </>
  );
};
