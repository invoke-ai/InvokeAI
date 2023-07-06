import { Flex, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineMultiSelect from 'common/components/IAIMantineMultiSelect';
import { forEach } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { useGetLoRAModelsQuery } from 'services/api/endpoints/models';
import { loraAdded } from '../store/loraSlice';
import IAIMantineSelectItemWithTooltip from '../../../common/components/IAIMantineSelectItemWithTooltip';

type LoraSelectItem = {
  label: string;
  value: string;
  description?: string;
};

const selector = createSelector(
  stateSelector,
  ({ lora }) => ({
    loras: lora.loras,
  }),
  defaultSelectorOptions
);

const ParamLoraSelect = () => {
  const dispatch = useAppDispatch();
  const { loras } = useAppSelector(selector);
  const { data: lorasQueryData } = useGetLoRAModelsQuery();

  const currentMainModel = useAppSelector(
    (state: RootState) => state.generation.model
  );

  const data = useMemo(() => {
    if (!lorasQueryData) {
      return [];
    }

    const data: LoraSelectItem[] = [];

    forEach(lorasQueryData.entities, (lora, id) => {
      if (!lora || Boolean(id in loras)) {
        return;
      }

      data.push({
        value: id,
        label: lora.name,
        description: 'This is a lora',
        ...(currentMainModel?.base_model !== lora.base_model
          ? { disabled: true, tooltip: 'Incompatible base model' }
          : {}),
      });
    });

    return data;
  }, [loras, lorasQueryData, currentMainModel?.base_model]);

  const handleChange = useCallback(
    (v: string[]) => {
      const loraEntity = lorasQueryData?.entities[v[0]];
      if (!loraEntity) {
        return;
      }
      v[0] && dispatch(loraAdded(loraEntity));
    },
    [dispatch, lorasQueryData?.entities]
  );

  if (lorasQueryData?.ids.length === 0) {
    return (
      <Flex sx={{ justifyContent: 'center', p: 2 }}>
        <Text sx={{ fontSize: 'sm', color: 'base.500', _dark: 'base.700' }}>
          No LoRAs Loaded
        </Text>
      </Flex>
    );
  }

  return (
    <IAIMantineMultiSelect
      placeholder={data.length === 0 ? 'All LoRAs added' : 'Add LoRA'}
      value={[]}
      data={data}
      maxDropdownHeight={400}
      nothingFound="No matching LoRAs"
      itemComponent={IAIMantineSelectItemWithTooltip}
      disabled={data.length === 0}
      filter={(value, selected, item: LoraSelectItem) =>
        item.label.toLowerCase().includes(value.toLowerCase().trim()) ||
        item.value.toLowerCase().includes(value.toLowerCase().trim())
      }
      onChange={handleChange}
    />
  );
};

export default ParamLoraSelect;
