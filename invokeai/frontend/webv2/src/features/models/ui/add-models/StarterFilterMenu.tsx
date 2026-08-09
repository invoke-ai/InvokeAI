import type { StarterModelFilters, StarterSortField } from '@features/models/core/starters';
import type { ModelTaxonomyType } from '@features/models/core/types';
import type { ModelFilterSortOption } from '@features/models/ui/shared/ModelFilterMenu';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ModelFilterMenu, SORT_FIELD_OPTIONS } from '@features/models/ui/shared/ModelFilterMenu';
import { useTranslation } from 'react-i18next';

const STARTER_SORT_FIELDS = SORT_FIELD_OPTIONS.filter(
  (option): option is ModelFilterSortOption<StarterSortField> => option.field !== 'size'
);

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
