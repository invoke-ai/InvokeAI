import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { IAISelectDataType } from 'common/components/IAIMantineSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { isEqual } from 'lodash-es';
import {
  selectAllSD1Models,
  selectByIdSD1Models,
} from './models/sd1ModelSlice';
import {
  selectAllSD2Models,
  selectByIdSD2Models,
} from './models/sd2ModelSlice';

export const modelSelector = createSelector(
  [(state: RootState) => state, generationSelector],
  (state, generation) => {
    let selectedModel = selectByIdSD1Models(state, generation.model);
    if (selectedModel === undefined)
      selectedModel = selectByIdSD2Models(state, generation.model);

    const sd1Models = selectAllSD1Models(state);
    const sd2Models = selectAllSD2Models(state);

    const sd1ModelDropDownData = selectAllSD1Models(state)
      .map<IAISelectDataType>((m) => ({
        value: m.name,
        label: m.name,
        group: '1.x Models',
      }))
      .sort((a, b) => a.label.localeCompare(b.label));

    const sd2ModelDropdownData = selectAllSD2Models(state)
      .map<IAISelectDataType>((m) => ({
        value: m.name,
        label: m.name,
        group: '2.x Models',
      }))
      .sort((a, b) => a.label.localeCompare(b.label));

    return {
      selectedModel,
      sd1Models,
      sd2Models,
      sd1ModelDropDownData,
      sd2ModelDropdownData,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);
