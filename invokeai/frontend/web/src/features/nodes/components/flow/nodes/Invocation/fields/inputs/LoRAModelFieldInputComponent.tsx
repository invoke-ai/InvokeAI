import { Flex, Text } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import { fieldLoRAModelValueChanged } from 'features/nodes/store/nodesSlice';
import {
  LoRAModelFieldInputTemplate,
  LoRAModelFieldInputInstance,
} from 'features/nodes/types/field';
import { FieldComponentProps } from './types';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToLoRAModelParam } from 'features/parameters/util/modelIdToLoRAModelParam';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useGetLoRAModelsQuery } from 'services/api/endpoints/models';
import { useTranslation } from 'react-i18next';

const LoRAModelFieldInputComponent = (
  props: FieldComponentProps<
    LoRAModelFieldInputInstance,
    LoRAModelFieldInputTemplate
  >
) => {
  const { nodeId, field } = props;
  const lora = field.value;
  const dispatch = useAppDispatch();
  const { data: loraModels } = useGetLoRAModelsQuery();
  const { t } = useTranslation();

  const data = useMemo(() => {
    if (!loraModels) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(loraModels.entities, (lora, id) => {
      if (!lora) {
        return;
      }

      data.push({
        value: id,
        label: lora.model_name,
        group: MODEL_TYPE_MAP[lora.base_model],
      });
    });

    return data.sort((a, b) => (a.disabled && !b.disabled ? 1 : -1));
  }, [loraModels]);

  const selectedLoRAModel = useMemo(
    () =>
      loraModels?.entities[`${lora?.base_model}/lora/${lora?.model_name}`] ??
      null,
    [loraModels?.entities, lora?.base_model, lora?.model_name]
  );

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newLoRAModel = modelIdToLoRAModelParam(v);

      if (!newLoRAModel) {
        return;
      }

      dispatch(
        fieldLoRAModelValueChanged({
          nodeId,
          fieldName: field.name,
          value: newLoRAModel,
        })
      );
    },
    [dispatch, field.name, nodeId]
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
          {t('models.noLoRAsLoaded')}
        </Text>
      </Flex>
    );
  }

  return (
    <IAIMantineSearchableSelect
      className="nowheel nodrag"
      value={selectedLoRAModel?.id ?? null}
      placeholder={
        data.length > 0 ? t('models.selectLoRA') : t('models.noLoRAsAvailable')
      }
      data={data}
      nothingFound={t('models.noMatchingLoRAs')}
      itemComponent={IAIMantineSelectItemWithTooltip}
      disabled={data.length === 0}
      filter={filterFunc}
      error={!selectedLoRAModel}
      onChange={handleChange}
      sx={{
        width: '100%',
        '.mantine-Select-dropdown': {
          width: '16rem !important',
        },
      }}
    />
  );
};

export default memo(LoRAModelFieldInputComponent);
