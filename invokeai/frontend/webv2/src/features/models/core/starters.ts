import type { ModelSortField } from './library';
import type { ModelTaxonomyType, StarterModel } from './types';

import { getModelBaseLabel } from './baseIdentity';

/** Pure filter/sort for the starter-models catalog; the installed-library twin is `library.ts`. */

export interface StarterModelFilters {
  /** null = all types. */
  typeFilter: ModelTaxonomyType | null;
  /** null = all bases. */
  baseFilter: string | null;
  sortField: ModelSortField;
  sortDirection: 'asc' | 'desc';
}

export const DEFAULT_STARTER_MODEL_FILTERS: StarterModelFilters = {
  baseFilter: null,
  sortDirection: 'asc',
  sortField: 'default',
  typeFilter: null,
};

interface IndexedStarterModel {
  index: number;
  model: StarterModel;
}

const compareStarterModels = (a: IndexedStarterModel, b: IndexedStarterModel, field: ModelSortField): number => {
  switch (field) {
    case 'default':
      return a.index - b.index;
    case 'name':
      return a.model.name.localeCompare(b.model.name, undefined, { sensitivity: 'base' });
    case 'base':
      return getModelBaseLabel(a.model.base).localeCompare(getModelBaseLabel(b.model.base));
    case 'format':
      return String(a.model.format ?? '').localeCompare(String(b.model.format ?? ''));
    case 'size':
      // Starter models have no local files, so size is not a meaningful sort.
      return 0;
  }
};

/** Filter by taxonomy + free-text terms, then sort; 'default' keeps catalog order. */
export const filterStarterModels = (
  models: readonly StarterModel[],
  filters: StarterModelFilters,
  searchTerm: string
): StarterModel[] => {
  const terms = searchTerm.toLowerCase().split(/\s+/).filter(Boolean);
  const directionFactor = filters.sortDirection === 'desc' ? -1 : 1;

  return models
    .map((model, index) => ({ index, model }))
    .filter(({ model }) => {
      const haystack =
        `${model.name} ${model.description} ${model.base} ${model.type} ${model.format ?? ''} ${model.variant ?? ''}`.toLowerCase();

      return (
        (filters.typeFilter === null || model.type === filters.typeFilter) &&
        (filters.baseFilter === null || model.base === filters.baseFilter) &&
        terms.every((term) => haystack.includes(term))
      );
    })
    .sort((a, b) => compareStarterModels(a, b, filters.sortField) * directionFactor)
    .map(({ model }) => model);
};
