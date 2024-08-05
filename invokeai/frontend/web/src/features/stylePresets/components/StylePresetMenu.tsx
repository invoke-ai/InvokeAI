import { Button, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { isModalOpenChanged, updatingStylePresetChanged } from 'features/stylePresets/store/slice';
import { useCallback } from 'react';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

import { StylePresetListItem } from './StylePresetListItem';

export const StylePresetMenu = () => {
  const { data } = useListStylePresetsQuery({});
  const dispatch = useAppDispatch();

  const handleClickAddNew = useCallback(() => {
    dispatch(updatingStylePresetChanged(null));
    dispatch(isModalOpenChanged(true));
  }, [dispatch]);

  return (
    <>
      <Flex flexDir="column" gap="2">
        <Flex alignItems="center" gap="10" w="full" justifyContent="space-between">
          <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
            Style Presets
          </Text>
          <Button size="sm" onClick={handleClickAddNew}>
            Add New
          </Button>
        </Flex>

        {data?.items.map((preset) => <StylePresetListItem preset={preset} key={preset.id} />)}
      </Flex>
    </>
  );
};
