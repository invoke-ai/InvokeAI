import { Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineMultiSelect from 'common/components/IAIMantineMultiSelect';
import { forEach } from 'lodash-es';
import { forwardRef, useCallback, useMemo } from 'react';
import { useGetLoRAModelsQuery } from 'services/api/endpoints/models';
import { loraAdded } from '../store/loraSlice';

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
        description: lora.description,
      });
    });

    return data;
  }, [loras, lorasQueryData]);

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

  return (
    <IAIMantineMultiSelect
      placeholder={data.length === 0 ? 'All LoRAs added' : 'Add LoRA'}
      value={[]}
      data={data}
      maxDropdownHeight={400}
      nothingFound="No matching LoRAs"
      itemComponent={SelectItem}
      disabled={data.length === 0}
      filter={(value, selected, item: LoraSelectItem) =>
        item.label.toLowerCase().includes(value.toLowerCase().trim()) ||
        item.value.toLowerCase().includes(value.toLowerCase().trim())
      }
      onChange={handleChange}
    />
  );
};

interface ItemProps extends React.ComponentPropsWithoutRef<'div'> {
  value: string;
  label: string;
  description?: string;
}

const SelectItem = forwardRef<HTMLDivElement, ItemProps>(
  ({ label, description, ...others }: ItemProps, ref) => {
    return (
      <div ref={ref} {...others}>
        <div>
          <Text>{label}</Text>
          {description && (
            <Text size="xs" color="base.600">
              {description}
            </Text>
          )}
        </div>
      </div>
    );
  }
);

SelectItem.displayName = 'SelectItem';

export default ParamLoraSelect;
