import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldSchedulerValueChanged } from 'features/nodes/store/nodesSlice';
import type { SchedulerFieldInputInstance, SchedulerFieldInputTemplate } from 'features/nodes/types/field';
import { SCHEDULER_OPTIONS } from 'features/parameters/types/constants';
import { isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<SchedulerFieldInputInstance, SchedulerFieldInputTemplate>;

const SchedulerFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterScheduler(v?.value)) {
        return;
      }
      dispatch(
        fieldSchedulerValueChanged({
          nodeId,
          fieldName: field.name,
          value: v.value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const value = useMemo(() => SCHEDULER_OPTIONS.find((o) => o.value === field?.value), [field?.value]);

  return (
    <FormControl className="nowheel nodrag">
      <Combobox value={value} options={SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(SchedulerFieldInputComponent);
