import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldSystemPromptValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { SystemPromptFieldInputInstance, SystemPromptFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListSystemPromptsQuery } from 'services/api/endpoints/systemPrompts';

import type { FieldComponentProps } from './types';

const SystemPromptFieldInputComponent = (
  props: FieldComponentProps<SystemPromptFieldInputInstance, SystemPromptFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { data: systemPrompts, isLoading } = useListSystemPromptsQuery();

  const options = useMemo<ComboboxOption[]>(() => {
    if (!systemPrompts) {
      return [];
    }
    return systemPrompts.map((p) => ({ label: p.name, value: p.id }));
  }, [systemPrompts]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      dispatch(
        fieldSystemPromptValueChanged({
          nodeId,
          fieldName: field.name,
          value: { system_prompt_id: v.value },
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
    return options.find((o) => o.value === _value.system_prompt_id) ?? null;
  }, [field.value, options]);

  const noOptionsMessage = useCallback(() => t('systemPrompts.noPromptsYet'), [t]);

  return (
    <Combobox
      className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
      value={value}
      options={options}
      onChange={onChange}
      placeholder={isLoading ? t('common.loading') : t('systemPrompts.selectSystemPrompt')}
      noOptionsMessage={noOptionsMessage}
    />
  );
};

export default memo(SystemPromptFieldInputComponent);
