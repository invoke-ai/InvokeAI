import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { fieldSchedulerValueChanged } from 'features/nodes/store/nodesSlice';
import {
  SchedulerFieldInputTemplate,
  SchedulerFieldInputInstance,
} from 'features/nodes/types/field';
import { FieldComponentProps } from './types';
import { ParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { SCHEDULER_LABEL_MAP } from 'features/parameters/types/constants';
import { map } from 'lodash-es';
import { memo, useCallback } from 'react';

const selector = createSelector(
  [stateSelector],
  ({ ui }) => {
    const { favoriteSchedulers: enabledSchedulers } = ui;

    const data = map(SCHEDULER_LABEL_MAP, (label, name) => ({
      value: name,
      label: label,
      group: enabledSchedulers.includes(name as ParameterScheduler)
        ? 'Favorites'
        : undefined,
    })).sort((a, b) => a.label.localeCompare(b.label));

    return {
      data,
    };
  },
  defaultSelectorOptions
);

const SchedulerFieldInputComponent = (
  props: FieldComponentProps<
    SchedulerFieldInputInstance,
    SchedulerFieldInputTemplate
  >
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data } = useAppSelector(selector);

  const handleChange = useCallback(
    (value: string | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldSchedulerValueChanged({
          nodeId,
          fieldName: field.name,
          value: value as ParameterScheduler,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <IAIMantineSearchableSelect
      className="nowheel nodrag"
      value={field.value}
      data={data}
      onChange={handleChange}
    />
  );
};

export default memo(SchedulerFieldInputComponent);
