import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOnChange } from 'common/components/InvSelect/types';
import { fieldSchedulerValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  SchedulerFieldInputInstance,
  SchedulerFieldInputTemplate,
} from 'features/nodes/types/field';
import { SCHEDULER_OPTIONS } from 'features/parameters/types/constants';
import { isParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<
  SchedulerFieldInputInstance,
  SchedulerFieldInputTemplate
>;

const SchedulerFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const onChange = useCallback<InvSelectOnChange>(
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

  const value = useMemo(
    () => SCHEDULER_OPTIONS.find((o) => o.value === field?.value),
    [field?.value]
  );

  return (
    <InvControl className="nowheel nodrag">
      <InvSelect
        value={value}
        options={SCHEDULER_OPTIONS}
        onChange={onChange}
      />
    </InvControl>
  );
};

export default memo(SchedulerFieldInputComponent);
