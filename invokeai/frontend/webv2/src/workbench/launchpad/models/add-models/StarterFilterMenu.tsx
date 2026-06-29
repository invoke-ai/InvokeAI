/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelFilterSortOption } from '@workbench/launchpad/models/shared/ModelFilterMenu';
import type { ModelSortField } from '@workbench/models/library';
import type { ModelTaxonomyType } from '@workbench/models/types';

import { ModelFilterMenu } from '@workbench/launchpad/models/shared/ModelFilterMenu';

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

const STARTER_SORT_FIELDS: ModelFilterSortOption[] = [
  { field: 'default', label: 'Default' },
  { field: 'name', label: 'Name' },
  { field: 'base', label: 'Base' },
  { field: 'format', label: 'Format' },
];

const isFiltering = (filters: StarterModelFilters): boolean =>
  filters.typeFilter !== null || filters.baseFilter !== null;

export const StarterFilterMenu = ({
  availableBases,
  availableTypes,
  filters,
  onChange,
}: {
  availableBases: string[];
  availableTypes: ModelTaxonomyType[];
  filters: StarterModelFilters;
  onChange: (filters: StarterModelFilters) => void;
}) => (
  <ModelFilterMenu
    ariaLabel="Filter and sort starter models"
    availableBases={availableBases}
    availableTypes={availableTypes}
    baseFilter={filters.baseFilter}
    isActive={isFiltering(filters)}
    sortDirection={filters.sortDirection}
    sortField={filters.sortField}
    sortFields={STARTER_SORT_FIELDS}
    typeFilter={filters.typeFilter}
    onBaseFilterChange={(baseFilter) => onChange({ ...filters, baseFilter })}
    onSortChange={(sortField, sortDirection) => onChange({ ...filters, sortDirection, sortField })}
    onTypeFilterChange={(typeFilter) => onChange({ ...filters, typeFilter })}
  />
);
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
