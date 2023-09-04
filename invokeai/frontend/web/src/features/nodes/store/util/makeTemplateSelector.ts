import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { AnyInvocationType } from 'services/events/types';

export const makeTemplateSelector = (type: AnyInvocationType) =>
  createSelector(
    stateSelector,
    ({ nodes }) => nodes.nodeTemplates[type],
    defaultSelectorOptions
  );
