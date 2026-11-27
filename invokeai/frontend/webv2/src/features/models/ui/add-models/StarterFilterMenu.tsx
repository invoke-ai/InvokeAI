import type { StarterModelFilters, StarterSortField } from '@features/models/core/starters';
import type { ModelTaxonomyType } from '@features/models/core/types';
import type { ModelFilterSortOption } from '@features/models/ui/shared/ModelFilterMenu';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ModelFilterMenu } from '@features/models/ui/shared/ModelFilterMenu';
import { useTranslation } from 'react-i18next';

/** Name is the starter list's default, so the catalog's own order is named for what it is, not "Default". */
const STARTER_SORT_FIELDS: readonly ModelFilterSortOption<StarterSortField>[] = [
  { field: 'default', labelKey: 'models.sort.catalog' },
  { field: 'name', labelKey: 'models.sort.name' },
  { field: 'base', labelKey: 'models.sort.base' },
  { field: 'format', labelKey: 'models.sort.format' },
];

const countActiveFilters = (filters: StarterModelFilters): number =>
  Number(filters.typeFilter !== null) + Number(filters.baseFilter !== null);

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
      activeFilterCount={countActiveFilters(filters)}
      ariaLabel={t('models.filterAndSortStarterModels')}
      availableBases={availableBases}
      availableTypes={availableTypes}
      baseFilter={filters.baseFilter}
      sortDirection={filters.sortDirection}
      sortField={filters.sortField}
      sortFields={STARTER_SORT_FIELDS}
      typeFilter={filters.typeFilter}
      onBaseFilterChange={(baseFilter) => onChange({ ...filters, baseFilter })}
      onResetFilters={() => onChange({ ...filters, baseFilter: null, typeFilter: null })}
      onSortChange={(sortField, sortDirection) => onChange({ ...filters, sortDirection, sortField })}
      onTypeFilterChange={(typeFilter) => onChange({ ...filters, typeFilter })}
    />
  );
};
