import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldStylePresetValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { StylePresetFieldInputInstance, StylePresetFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

import type { FieldComponentProps } from './types';

const StylePresetFieldInputComponent = (
  props: FieldComponentProps<StylePresetFieldInputInstance, StylePresetFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { data: stylePresets, isLoading } = useListStylePresetsQuery();

  const options = useMemo<ComboboxOption[]>(() => {
    const _options: ComboboxOption[] = [];
    if (stylePresets) {
      for (const preset of stylePresets) {
        _options.push({
          label: preset.name,
          value: preset.id,
        });
      }
    }
    return _options;
  }, [stylePresets]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        return;
      }

      dispatch(
        fieldStylePresetValueChanged({
          nodeId,
          fieldName: field.name,
          value: { style_preset_id: v.value },
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const value = useMemo(() => {
    const _value = field.value;
    if (!_value) {
      return null;
    }
    return options.find((o) => o.value === _value.style_preset_id) ?? null;
  }, [field.value, options]);

  const noOptionsMessage = useCallback(() => t('stylePresets.noMatchingPresets'), [t]);

  return (
    <Combobox
      className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
      value={value}
      options={options}
      onChange={onChange}
      placeholder={isLoading ? t('common.loading') : t('stylePresets.selectPreset')}
      noOptionsMessage={noOptionsMessage}
    />
  );
};

export default memo(StylePresetFieldInputComponent);
