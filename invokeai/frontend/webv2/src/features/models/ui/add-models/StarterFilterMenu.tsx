import type { ModelSortField } from '@features/models/core/library';
import type { ModelTaxonomyType } from '@features/models/core/types';
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelFilterSortOption } from '@features/models/ui/shared/ModelFilterMenu';

import { ModelFilterMenu } from '@features/models/ui/shared/ModelFilterMenu';
import { useTranslation } from 'react-i18next';

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
  { field: 'default', labelKey: 'models.sortDefault' },
  { field: 'name', labelKey: 'models.sortName' },
  { field: 'base', labelKey: 'models.sortBase' },
  { field: 'format', labelKey: 'models.sortFormat' },
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
}) => {
  const { t } = useTranslation();

  return (
    <ModelFilterMenu
      ariaLabel={t('models.filterAndSortStarterModels')}
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
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
