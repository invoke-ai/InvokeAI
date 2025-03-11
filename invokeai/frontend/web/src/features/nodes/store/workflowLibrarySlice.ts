import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import { atom, computed } from 'nanostores';
import type { SQLiteDirection, WorkflowRecordOrderBy } from 'services/api/types';

type WorkflowLibraryState = {
  searchTerm: string;
  orderBy: WorkflowRecordOrderBy;
  direction: SQLiteDirection;
  tags: string[];
  categories: WorkflowCategory[];
  showOpenedWorkflowsOnly: boolean;
};

const initialWorkflowLibraryState: WorkflowLibraryState = {
  searchTerm: '',
  orderBy: 'opened_at',
  direction: 'DESC',
  tags: [],
  categories: ['user'],
  showOpenedWorkflowsOnly: false,
};

export const workflowLibrarySlice = createSlice({
  name: 'workflowLibrary',
  initialState: initialWorkflowLibraryState,
  reducers: {
    workflowLibrarySearchTermChanged: (state, action: PayloadAction<string>) => {
      state.searchTerm = action.payload;
    },
    workflowLibraryOrderByChanged: (state, action: PayloadAction<WorkflowRecordOrderBy>) => {
      state.orderBy = action.payload;
    },
    workflowLibraryDirectionChanged: (state, action: PayloadAction<SQLiteDirection>) => {
      state.direction = action.payload;
    },
    workflowLibraryCategoriesChanged: (state, action: PayloadAction<WorkflowCategory[]>) => {
      state.categories = action.payload;
      state.searchTerm = '';
    },
    workflowLibraryShowOpenedWorkflowsOnlyChanged: (state, action: PayloadAction<boolean>) => {
      state.showOpenedWorkflowsOnly = action.payload;
    },
    workflowLibraryTagToggled: (state, action: PayloadAction<string>) => {
      const tag = action.payload;
      const tags = state.tags;
      if (tags.includes(tag)) {
        state.tags = tags.filter((t) => t !== tag);
      } else {
        state.tags = [...tags, tag];
      }
    },
    workflowLibraryTagsReset: (state) => {
      state.tags = [];
    },
  },
});

export const {
  workflowLibrarySearchTermChanged,
  workflowLibraryOrderByChanged,
  workflowLibraryDirectionChanged,
  workflowLibraryCategoriesChanged,
  workflowLibraryShowOpenedWorkflowsOnlyChanged,
  workflowLibraryTagToggled,
  workflowLibraryTagsReset,
} = workflowLibrarySlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateWorkflowLibraryState = (state: any): any => state;

export const workflowLibraryPersistConfig: PersistConfig<WorkflowLibraryState> = {
  name: workflowLibrarySlice.name,
  initialState: initialWorkflowLibraryState,
  migrate: migrateWorkflowLibraryState,
  persistDenylist: [],
};

const selectWorkflowLibrarySlice = (state: RootState) => state.workflowLibrary;
const createWorkflowLibrarySelector = <T>(selector: Selector<WorkflowLibraryState, T>) =>
  createSelector(selectWorkflowLibrarySlice, selector);

export const selectWorkflowLibrarySearchTerm = createWorkflowLibrarySelector(({ searchTerm }) => searchTerm);
export const selectWorkflowLibraryHasSearchTerm = createWorkflowLibrarySelector(({ searchTerm }) => !!searchTerm);
export const selectWorkflowLibraryOrderBy = createWorkflowLibrarySelector(({ orderBy }) => orderBy);
export const selectWorkflowLibraryDirection = createWorkflowLibrarySelector(({ direction }) => direction);
export const selectWorkflowLibraryTags = createWorkflowLibrarySelector(({ tags }) => tags);
export const selectWorkflowLibraryCategories = createWorkflowLibrarySelector(({ categories }) => categories);
export const selectWorkflowLibraryShowOpenedWorkflowsOnly = createWorkflowLibrarySelector(({ showOpenedWorkflowsOnly }) => showOpenedWorkflowsOnly);

export const DEFAULT_WORKFLOW_LIBRARY_CATEGORIES = ['user', 'default'] satisfies WorkflowCategory[];
export const $workflowLibraryCategoriesOptions = atom<WorkflowCategory[]>(DEFAULT_WORKFLOW_LIBRARY_CATEGORIES);

export type WorkflowTagCategory = { categoryTKey: string; tags: string[] };
export const DEFAULT_WORKFLOW_LIBRARY_TAG_CATEGORIES: WorkflowTagCategory[] = [
  { categoryTKey: 'Industry', tags: ['Architecture', 'Fashion', 'Game Dev', 'Food'] },
  { categoryTKey: 'Common Tasks', tags: ['Upscaling', 'Text to Image', 'Image to Image'] },
  { categoryTKey: 'Model Architecture', tags: ['SD1.5', 'SDXL', 'Bria', 'FLUX'] },
  { categoryTKey: 'Tech Showcase', tags: ['Control', 'Reference Image'] },
];
export const $workflowLibraryTagCategoriesOptions = atom<WorkflowTagCategory[]>(
  DEFAULT_WORKFLOW_LIBRARY_TAG_CATEGORIES
);
export const $workflowLibraryTagOptions = computed($workflowLibraryTagCategoriesOptions, (tagCategories) =>
  tagCategories.flatMap(({ tags }) => tags)
);
