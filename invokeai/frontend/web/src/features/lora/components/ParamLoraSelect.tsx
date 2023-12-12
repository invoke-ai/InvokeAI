import { Flex, Text } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { RootState, stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import { loraAdded } from 'features/lora/store/loraSlice';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetLoRAModelsQuery } from 'services/api/endpoints/models';

const selector = createMemoizedSelector(stateSelector, ({ lora }) => ({
  loras: lora.loras,
}));

const ParamLoRASelect = () => {
  const dispatch = useAppDispatch();
  const { loras } = useAppSelector(selector);
  const { data: loraModels } = useGetLoRAModelsQuery();
  const { t } = useTranslation();
  const currentMainModel = useAppSelector(
    (state: RootState) => state.generation.model
  );

  const data = useMemo(() => {
    if (!loraModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(loraModels.entities, (lora, id) => {
      if (!lora || Boolean(id in loras)) {
        return;
      }

      const disabled = currentMainModel?.base_model !== lora.base_model;

      data.push({
        value: id,
        label: lora.model_name,
        disabled,
        group: MODEL_TYPE_MAP[lora.base_model],
        tooltip: disabled
          ? `Incompatible base model: ${lora.base_model}`
          : undefined,
      });
    });

    data.sort((a, b) => (a.label && !b.label ? 1 : -1));

    return data.sort((a, b) => (a.disabled && !b.disabled ? 1 : -1));
  }, [loras, loraModels, currentMainModel?.base_model]);

  const handleChange = useCallback(
    (v: string | null | undefined) => {
      if (!v) {
        return;
      }
      const loraEntity = loraModels?.entities[v];

      if (!loraEntity) {
        return;
      }

      dispatch(loraAdded(loraEntity));
    },
    [dispatch, loraModels?.entities]
  );

  const filterFunc = useCallback(
    (value: string, item: SelectItem) =>
      item.label?.toLowerCase().includes(value.toLowerCase().trim()) ||
      item.value.toLowerCase().includes(value.toLowerCase().trim()),
    []
  );

  if (loraModels?.ids.length === 0) {
    return (
      <Flex sx={{ justifyContent: 'center', p: 2 }}>
        <Text sx={{ fontSize: 'sm', color: 'base.500', _dark: 'base.700' }}>
          {t('models.noLoRAsInstalled')}
        </Text>
      </Flex>
    );
  }

  return (
    <IAIMantineSearchableSelect
      placeholder={data.length === 0 ? 'All LoRAs added' : t('models.addLora')}
      value={null}
      data={data}
      nothingFound="No matching LoRAs"
      itemComponent={IAIMantineSelectItemWithTooltip}
      disabled={data.length === 0}
      filter={filterFunc}
      onChange={handleChange}
      data-testid="add-lora"
    />
  );
};

export default memo(ParamLoRASelect);
