import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController, useWatch } from 'react-hook-form';
import type { AnyModelConfig } from 'services/api/types';
import { useGetVaeModelsQuery } from '../../../../../services/api/endpoints/models';
import { useTranslation } from 'react-i18next';
import { GroupBase } from 'chakra-react-select';
import { map, reduce, groupBy } from 'lodash-es';

const VaeSelect = (props: UseControllerProps<AnyModelConfig>) => {
  const { t } = useTranslation();
  const { field } = useController(props);
  const { data } = useGetVaeModelsQuery();
  const base = useWatch({ control: props.control, name: 'base' });

  const onChange = useCallback<ComboboxOnChange>(
    (value) => {
      if (!value) {
        field.onChange(null);
        return;
      }

      field.onChange(value.value);
    },
    [field]
  );

  const options = useMemo<GroupBase<ComboboxOption>[]>(() => {
    if (!data) {
      return [];
    }
    const modelEntitiesArray = map(data.entities);
    const groupedModels = groupBy(modelEntitiesArray, 'base');
    const _options = reduce(
      groupedModels,
      (acc, val, label) => {
        acc.push({
          label,
          options: val.map((model) => ({
            label: model.name,
            value: model.path,
            isDisabled: base !== model.base,
          })),
        });
        return acc;
      },
      [] as GroupBase<ComboboxOption>[]
    );
    _options.sort((a) => (a.label === base ? -1 : 1));
    return _options;
  }, [data, base]);

  const value = useMemo(
    () => options.flatMap((o) => o.options).find((m) => (field.value ? m.value === field.value : false)) ?? null,
    [options, field.value]
  );

  return (
    <Combobox
      isClearable
      value={value}
      options={options}
      onChange={onChange}
      placeholder={value ? value.value : t('models.defaultVAE')}
    />
  );
};

export default typedMemo(VaeSelect);
