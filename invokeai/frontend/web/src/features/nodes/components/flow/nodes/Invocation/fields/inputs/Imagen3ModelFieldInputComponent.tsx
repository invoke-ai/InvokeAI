import { useAppDispatch } from 'app/store/storeHooks';
import { ModelFieldCombobox } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelFieldCombobox';
import { fieldImagen3ModelValueChanged } from 'features/nodes/store/nodesSlice';
import type { Imagen3ModelFieldInputInstance, Imagen3ModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useImagen3Models } from 'services/api/hooks/modelsByType';
import type { ApiModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const Imagen3ModelFieldInputComponent = (
  props: FieldComponentProps<Imagen3ModelFieldInputInstance, Imagen3ModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useImagen3Models();

  const onChange = useCallback(
    (value: ApiModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldImagen3ModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <ModelFieldCombobox
      value={field.value}
      modelConfigs={modelConfigs}
      isLoadingConfigs={isLoading}
      onChange={onChange}
      required={props.fieldTemplate.required}
    />
  );
};

export default memo(Imagen3ModelFieldInputComponent);
