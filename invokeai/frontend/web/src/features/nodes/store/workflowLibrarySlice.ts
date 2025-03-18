import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import type { WorkflowMode } from 'features/nodes/store/types';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import { atom, computed } from 'nanostores';
import type { SQLiteDirection, WorkflowRecordOrderBy } from 'services/api/types';

export type WorkflowLibraryView = 'recent' | 'yours' | 'private' | 'shared' | 'defaults' | 'published';

type WorkflowLibraryState = {
  mode: WorkflowMode;
  view: WorkflowLibraryView;
  orderBy: WorkflowRecordOrderBy;
  direction: SQLiteDirection;
  searchTerm: string;
  selectedTags: string[];
};

const initialWorkflowLibraryState: WorkflowLibraryState = {
  mode: 'view',
  searchTerm: '',
  orderBy: 'opened_at',
  direction: 'DESC',
  selectedTags: [],
  view: 'defaults',
};

export const workflowLibrarySlice = createSlice({
  name: 'workflowLibrary',
  initialState: initialWorkflowLibraryState,
  reducers: {
    workflowModeChanged: (state, action: PayloadAction<WorkflowMode>) => {
      state.mode = action.payload;
    },
    workflowLibrarySearchTermChanged: (state, action: PayloadAction<string>) => {
      state.searchTerm = action.payload;
    },
    workflowLibraryOrderByChanged: (state, action: PayloadAction<WorkflowRecordOrderBy>) => {
      state.orderBy = action.payload;
    },
    workflowLibraryDirectionChanged: (state, action: PayloadAction<SQLiteDirection>) => {
      state.direction = action.payload;
    },
    workflowLibraryViewChanged: (state, action: PayloadAction<WorkflowLibraryState['view']>) => {
      state.view = action.payload;
      state.searchTerm = '';
      if (action.payload === 'recent') {
        state.orderBy = 'opened_at';
        state.direction = 'DESC';
      }
    },
    workflowLibraryTagToggled: (state, action: PayloadAction<string>) => {
      const tag = action.payload;
      const tags = state.selectedTags;
      if (tags.includes(tag)) {
        state.selectedTags = tags.filter((t) => t !== tag);
      } else {
        state.selectedTags = [...tags, tag];
      }
    },
    workflowLibraryTagsReset: (state) => {
      state.selectedTags = [];
    },
  },
});

export const {
  workflowModeChanged,
  workflowLibrarySearchTermChanged,
  workflowLibraryOrderByChanged,
  workflowLibraryDirectionChanged,
  workflowLibraryTagToggled,
  workflowLibraryTagsReset,
  workflowLibraryViewChanged,
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

export const selectWorkflowMode = createWorkflowLibrarySelector((workflow) => workflow.mode);
export const selectWorkflowLibrarySearchTerm = createWorkflowLibrarySelector(({ searchTerm }) => searchTerm);
export const selectWorkflowLibraryHasSearchTerm = createWorkflowLibrarySelector(({ searchTerm }) => !!searchTerm);
export const selectWorkflowLibraryOrderBy = createWorkflowLibrarySelector(({ orderBy }) => orderBy);
export const selectWorkflowLibraryDirection = createWorkflowLibrarySelector(({ direction }) => direction);
export const selectWorkflowLibrarySelectedTags = createWorkflowLibrarySelector(({ selectedTags }) => selectedTags);
export const selectWorkflowLibraryView = createWorkflowLibrarySelector(({ view }) => view);

export const DEFAULT_WORKFLOW_LIBRARY_CATEGORIES = ['user', 'default'] satisfies WorkflowCategory[];
export const $workflowLibraryCategoriesOptions = atom<WorkflowCategory[]>(DEFAULT_WORKFLOW_LIBRARY_CATEGORIES);

export type WorkflowTagCategory = { categoryTKey: string; tags: Array<{ label: string; recommended?: boolean }> };
export const DEFAULT_WORKFLOW_LIBRARY_TAG_CATEGORIES: WorkflowTagCategory[] = [
  {
    categoryTKey: 'Industry',
    tags: [{ label: 'Architecture' }, { label: 'Fashion' }, { label: 'Game Dev' }, { label: 'Food' }],
  },
  {
    categoryTKey: 'Common Tasks',
    tags: [{ label: 'Upscaling' }, { label: 'Text to Image' }, { label: 'Image to Image' }],
  },
  {
    categoryTKey: 'Model Architecture',
    tags: [{ label: 'SD1.5' }, { label: 'SDXL' }, { label: 'SD3.5' }, { label: 'FLUX' }, { label: 'CogView4' }],
  },
  { categoryTKey: 'Tech Showcase', tags: [{ label: 'Control' }, { label: 'Reference Image' }] },
];
export const $workflowLibraryTagCategoriesOptions = atom<WorkflowTagCategory[]>(
  DEFAULT_WORKFLOW_LIBRARY_TAG_CATEGORIES
);
export const $workflowLibraryTagOptions = computed($workflowLibraryTagCategoriesOptions, (tagCategories) =>
  tagCategories.flatMap(({ tags }) => tags)
);

export type WorkflowSortOption = 'opened_at' | 'created_at' | 'updated_at' | 'name';
export const DEFAULT_WORKFLOW_LIBRARY_SORT_OPTIONS: WorkflowSortOption[] = [
  'opened_at',
  'created_at',
  'updated_at',
  'name',
];
export const $workflowLibrarySortOptions = atom<WorkflowSortOption[]>(DEFAULT_WORKFLOW_LIBRARY_SORT_OPTIONS);
