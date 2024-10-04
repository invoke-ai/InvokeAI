import { Badge, ConfirmationAlertDialog, Flex, IconButton, Text, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import {
  $isStylePresetsMenuOpen,
  activeStylePresetIdChanged,
  selectStylePresetActivePresetId,
} from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold, PiPencilBold, PiTrashBold } from 'react-icons/pi';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';
import { useDeleteStylePresetMutation } from 'services/api/endpoints/stylePresets';

import StylePresetImage from './StylePresetImage';

export const StylePresetListItem = ({ preset }: { preset: StylePresetRecordWithImage }) => {
  const dispatch = useAppDispatch();
  const [deleteStylePreset] = useDeleteStylePresetMutation();
  const activeStylePresetId = useAppSelector(selectStylePresetActivePresetId);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();

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
      onOpen();
    },
    [onOpen]
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

  const handleDeletePreset = useCallback(async () => {
    try {
      await deleteStylePreset(preset.id).unwrap();
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
        gap={4}
        onPointerUp={handleClickApply}
        cursor="pointer"
        _hover={{ backgroundColor: 'base.750' }}
        py={3}
        px={2}
        borderRadius="base"
        alignItems="flex-start"
        w="full"
      >
        <StylePresetImage presetImageUrl={preset.image} />
        <Flex flexDir="column" w="full">
          <Flex w="full" justifyContent="space-between" alignItems="flex-start">
            <Flex alignItems="center" gap={2}>
              <Text fontSize="md" noOfLines={2}>
                {preset.name}
              </Text>
              {activeStylePresetId === preset.id && (
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

            <Flex alignItems="center" gap={1}>
              <IconButton
                size="sm"
                variant="ghost"
                aria-label={t('stylePresets.copyTemplate')}
                onPointerUp={handleClickCopy}
                icon={<PiCopyBold />}
              />
              {preset.type !== 'default' && (
                <>
                  <IconButton
                    size="sm"
                    variant="ghost"
                    aria-label={t('stylePresets.editTemplate')}
                    onPointerUp={handleClickEdit}
                    icon={<PiPencilBold />}
                  />
                  <IconButton
                    size="sm"
                    variant="ghost"
                    aria-label={t('stylePresets.deleteTemplate')}
                    onPointerUp={handleClickDelete}
                    colorScheme="error"
                    icon={<PiTrashBold />}
                  />
                </>
              )}
            </Flex>
          </Flex>

          <Flex flexDir="column" gap={1}>
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
        acceptButtonText={t('common.delete')}
        cancelButtonText={t('common.cancel')}
        useInert={false}
      >
        <p>{t('stylePresets.deleteTemplate2')}</p>
      </ConfirmationAlertDialog>
    </>
  );
};
