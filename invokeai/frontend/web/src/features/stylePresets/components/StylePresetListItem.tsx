import { Badge, Button, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDeleteStylePreset } from 'features/stylePresets/components/DeleteStylePresetDialog';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import {
  $isStylePresetsMenuOpen,
  activeStylePresetIdChanged,
  selectShowPromptPreviews,
  selectStylePresetActivePresetId,
} from 'features/stylePresets/store/stylePresetSlice';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold, PiPencilBold, PiTrashBold } from 'react-icons/pi';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';

import StylePresetImage from './StylePresetImage';

export const StylePresetListItem = ({ preset }: { preset: StylePresetRecordWithImage }) => {
  const dispatch = useAppDispatch();
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);
  const showPromptPreviews = useAppSelector(selectShowPromptPreviews);
  const { t } = useTranslation();
  const deleteStylePreset = useDeleteStylePreset();

  const handleClickEdit = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      const { name, preset_data } = preset;
      const { positive_prompt, negative_prompt } = preset_data;

      $stylePresetModalState.set({
        prefilledFormData: {
          name,
          positivePrompt: positive_prompt || '',
          negativePrompt: negative_prompt || '',
          imageUrl: preset.image,
          type: preset.type,
        },
        updatingStylePresetId: preset.id,
        isModalOpen: true,
      });
    },
    [preset]
  );

  const handleClickApply = useCallback(() => {
    dispatch(activeStylePresetIdChanged(preset.id));
    $isStylePresetsMenuOpen.set(false);
  }, [dispatch, preset.id]);

  const handleClickDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      deleteStylePreset(preset);
    },
    [deleteStylePreset, preset]
  );

  const handleClickCopy = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      const { name, preset_data } = preset;
      const { positive_prompt, negative_prompt } = preset_data;

      $stylePresetModalState.set({
        prefilledFormData: {
          name: `${name} (${t('common.copy')})`,
          positivePrompt: positive_prompt || '',
          negativePrompt: negative_prompt || '',
          imageUrl: preset.image,
          type: 'user',
        },
        updatingStylePresetId: null,
        isModalOpen: true,
      });
    },
    [preset, t]
  );

  return (
    <Button
      as={Flex}
      role="button"
      gap={3}
      onClick={handleClickApply}
      p={3}
      h="unset"
      variant="ghost"
      w="full"
      alignItems="flex-start"
    >
      <StylePresetImage presetImageUrl={preset.image} />
      <Flex flexDir="column" w="full" alignItems="flex-start" flexGrow={1} minW={0} gap={2}>
        <Flex gap={2} w="full" justifyContent="space-between" alignItems="center" minW={0} minH={8}>
          <Text fontSize="md" noOfLines={2} fontWeight="semibold" color="base.100">
            {preset.name}
          </Text>
          {activeStylePresetId === preset.id && (
            <Badge color="invokeBlue.400" borderColor="invokeBlue.700" borderWidth={1} bg="transparent" flexShrink={0}>
              {t('stylePresets.active')}
            </Badge>
          )}
          <Spacer />
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            aria-label={t('stylePresets.copyTemplate')}
            onClick={handleClickCopy}
            icon={<PiCopyBold />}
          />
          {preset.type !== 'default' && (
            <>
              <IconButton
                size="sm"
                variant="link"
                alignSelf="stretch"
                aria-label={t('stylePresets.editTemplate')}
                onClick={handleClickEdit}
                icon={<PiPencilBold />}
              />
              <IconButton
                size="sm"
                variant="link"
                alignSelf="stretch"
                aria-label={t('stylePresets.deleteTemplate')}
                onClick={handleClickDelete}
                colorScheme="error"
                icon={<PiTrashBold />}
              />
            </>
          )}
        </Flex>
        {showPromptPreviews && (
          <>
            <Flex gap={1} minW={0} fontSize="sm" whiteSpace="normal">
              <Text>
                {t('stylePresets.positivePrompt')}:{' '}
                <Text as="span" fontWeight="normal">
                  {preset.preset_data.positive_prompt}
                </Text>
              </Text>
            </Flex>
            <Flex gap={1} minW={0} fontSize="sm" whiteSpace="normal">
              <Text>
                {t('stylePresets.negativePrompt')}:{' '}
                <Text as="span" fontWeight="normal">
                  {preset.preset_data.negative_prompt}
                </Text>
              </Text>
            </Flex>
          </>
        )}
      </Flex>
    </Button>
  );
};
