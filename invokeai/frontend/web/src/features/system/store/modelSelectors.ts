import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { reduce } from 'lodash';

export const modelSelector = (state: RootState) => state.models;
