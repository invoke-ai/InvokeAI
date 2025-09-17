import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldModelIdentifierValueChanged } from 'features/nodes/store/nodesSlice';
import type { ModelIdentifierFieldInputInstance, ModelIdentifierFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<ModelIdentifierFieldInputInstance, ModelIdentifierFieldInputTemplate>;

const ModelIdentifierFieldInputComponent = (props: Props) => {
  const { nodeId, field, fieldTemplate } = props;
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetModelConfigsQuery();
  const onChange = useCallback(
    (value: AnyModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldModelIdentifierValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const modelConfigs = useMemo(() => {
    if (!data) {
      return EMPTY_ARRAY;
    }

    if (!fieldTemplate.ui_model_base && !fieldTemplate.ui_model_type) {
      return modelConfigsAdapterSelectors.selectAll(data);
    }

    return modelConfigsAdapterSelectors.selectAll(data).filter((config) => {
      if (fieldTemplate.ui_model_base && !fieldTemplate.ui_model_base.includes(config.base)) {
        return false;
      }
      if (fieldTemplate.ui_model_type && !fieldTemplate.ui_model_type.includes(config.type)) {
        return false;
      }
      if (
        fieldTemplate.ui_model_variant &&
        'variant' in config &&
        config.variant &&
        !fieldTemplate.ui_model_variant.includes(config.variant)
      ) {
        return false;
      }
      return true;
    });
  }, [data, fieldTemplate.ui_model_base, fieldTemplate.ui_model_type, fieldTemplate.ui_model_variant]);

  return (
    <ModelFieldCombobox
      value={field.value}
      modelConfigs={modelConfigs}
      isLoadingConfigs={isLoading}
      onChange={onChange}
      required={props.fieldTemplate.required}
      groupByType
    />
  );
};

export default memo(ModelIdentifierFieldInputComponent);
