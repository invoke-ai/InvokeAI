import { Badge, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import { isModalOpenChanged, updatingStylePresetChanged } from 'features/stylePresets/store/stylePresetModalSlice';
import { activeStylePresetChanged, isMenuOpenChanged } from 'features/stylePresets/store/stylePresetSlice';
import { useCallback } from 'react';
import { PiPencilBold, PiTrashBold } from 'react-icons/pi';
import type { StylePresetRecordDTO } from 'services/api/endpoints/stylePresets';
import { useDeleteStylePresetMutation } from 'services/api/endpoints/stylePresets';

export const StylePresetListItem = ({ preset }: { preset: StylePresetRecordDTO }) => {
  const dispatch = useAppDispatch();
  const [deleteStylePreset] = useDeleteStylePresetMutation();
  const activeStylePreset = useAppSelector((s) => s.stylePreset.activeStylePreset);

  const handleClickEdit = useCallback(() => {
    dispatch(updatingStylePresetChanged(preset));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, preset]);

  const handleClickApply = useCallback(() => {
    dispatch(activeStylePresetChanged(preset));
    dispatch(isMenuOpenChanged(false));
  }, [dispatch, preset]);

  const handleDeletePreset = useCallback(async () => {
    try {
      await deleteStylePreset(preset.id);
    } catch (error) {}
  }, [preset]);

  return (
    <Flex
      gap="4"
      onClick={handleClickApply}
      cursor="pointer"
      _hover={{ backgroundColor: 'base.750' }}
      padding="10px"
      borderRadius="base"
      alignItems="center"
      w="full"
    >
      <ModelImage image_url={null} />
      <Flex flexDir="column" w="full">
        <Flex w="full" justifyContent="space-between">
          <Flex alignItems="center" gap="2">
            <Text fontSize="md">{preset.name}</Text>
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
            <IconButton size="sm" variant="ghost" aria-label="Edit" onClick={handleClickEdit} icon={<PiPencilBold />} />
            <IconButton
              size="sm"
              variant="ghost"
              aria-label="Delete"
              onClick={handleDeletePreset}
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
  );
};
