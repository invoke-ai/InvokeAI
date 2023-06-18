import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { IAISelectDataType } from 'common/components/IAIMantineSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { isEqual } from 'lodash-es';

import {
  selectAllSD1PipelineModels,
  selectByIdSD1PipelineModels,
} from './models/sd1PipelineModelSlice';

import {
  selectAllSD2PipelineModels,
  selectByIdSD2PipelineModels,
} from './models/sd2PipelineModelSlice';

export const modelSelector = createSelector(
  [(state: RootState) => state, generationSelector],
  (state, generation) => {
    let selectedModel = selectByIdSD1PipelineModels(state, generation.model);
    if (selectedModel === undefined)
      selectedModel = selectByIdSD2PipelineModels(state, generation.model);

    const sd1PipelineModels = selectAllSD1PipelineModels(state);
    const sd2PipelineModels = selectAllSD2PipelineModels(state);

    const allPipelineModels = sd1PipelineModels.concat(sd2PipelineModels);

    const sd1PipelineModelDropDownData = selectAllSD1PipelineModels(state)
      .map<IAISelectDataType>((m) => ({
        value: m.name,
        label: m.name,
        group: '1.x Models',
      }))
      .sort((a, b) => a.label.localeCompare(b.label));

    const sd2PipelineModelDropdownData = selectAllSD2PipelineModels(state)
      .map<IAISelectDataType>((m) => ({
        value: m.name,
        label: m.name,
        group: '2.x Models',
      }))
      .sort((a, b) => a.label.localeCompare(b.label));

    return {
      selectedModel,
      allPipelineModels,
      sd1PipelineModels,
      sd2PipelineModels,
      sd1PipelineModelDropDownData,
      sd2PipelineModelDropdownData,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);
