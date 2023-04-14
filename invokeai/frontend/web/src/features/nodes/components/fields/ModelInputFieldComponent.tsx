import { Select } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import { ModelInputField } from 'features/nodes/types';
import {
  selectModelsById,
  selectModelsIds,
} from 'features/system/store/modelSlice';
import { isEqual, map } from 'lodash';
import { ChangeEvent } from 'react';
import { FieldComponentProps } from './types';

const availableModelsSelector = createSelector(
  [selectModelsIds],
  (allModelNames) => {
    return { allModelNames };
    // return map(modelList, (_, name) => name);
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const ModelInputFieldComponent = (
  props: FieldComponentProps<ModelInputField>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const { allModelNames } = useAppSelector(availableModelsSelector);

  const handleValueChanged = (e: ChangeEvent<HTMLSelectElement>) => {
    dispatch(
      fieldValueChanged({
        nodeId,
        fieldId: field.name,
        value: e.target.value,
      })
    );
  };

  return (
    <Select onChange={handleValueChanged} value={field.value}>
      {allModelNames.map((option) => (
        <option key={option}>{option}</option>
      ))}
    </Select>
  );
};
