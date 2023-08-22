import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { fieldSchedulerValueChanged } from 'features/nodes/store/nodesSlice';
import {
  SchedulerInputFieldTemplate,
  SchedulerInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import {
  SCHEDULER_LABEL_MAP,
  SchedulerParam,
} from 'features/parameters/types/parameterSchemas';
import { map } from 'lodash-es';
import { memo, useCallback } from 'react';

const selector = createSelector(
  [stateSelector],
  ({ ui }) => {
    const { favoriteSchedulers: enabledSchedulers } = ui;

    const data = map(SCHEDULER_LABEL_MAP, (label, name) => ({
      value: name,
      label: label,
      group: enabledSchedulers.includes(name as SchedulerParam)
        ? 'Favorites'
        : undefined,
    })).sort((a, b) => a.label.localeCompare(b.label));

    return {
      data,
    };
  },
  defaultSelectorOptions
);

const SchedulerInputField = (
  props: FieldComponentProps<
    SchedulerInputFieldValue,
    SchedulerInputFieldTemplate
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
          value: value as SchedulerParam,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <IAIMantineSearchableSelect
      className="nowheel nodrag"
      sx={{
        '.mantine-Select-dropdown': {
          width: '14rem !important',
        },
      }}
      value={field.value}
      data={data}
      onChange={handleChange}
    />
  );
};

export default memo(SchedulerInputField);
