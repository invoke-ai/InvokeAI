import { Button, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { isModalOpenChanged, updatingStylePresetChanged } from 'features/stylePresets/store/stylePresetModalSlice';
import { useCallback } from 'react';
import type { StylePresetRecordDTO } from 'services/api/endpoints/stylePresets';
import { useDeleteStylePresetMutation } from 'services/api/endpoints/stylePresets';
import { activeStylePresetChanged, isMenuOpenChanged } from '../store/stylePresetSlice';

export const StylePresetListItem = ({ preset }: { preset: StylePresetRecordDTO }) => {
  const dispatch = useAppDispatch();
  const [deleteStylePreset] = useDeleteStylePresetMutation();

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
    <>
      <Flex flexDir="column" gap="2">
        <Text fontSize="md">{preset.name}</Text>
        <Flex flexDir="column" layerStyle="third" borderRadius="base" padding="10px">
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
          <Button onClick={handleClickEdit}>Edit</Button>
          <Button onClick={handleDeletePreset}>Delete</Button>
          <Button onClick={handleClickApply}>Apply</Button>
        </Flex>
      </Flex>
    </>
  );
};
