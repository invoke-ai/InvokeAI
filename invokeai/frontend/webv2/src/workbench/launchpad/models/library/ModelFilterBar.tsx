/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelLibraryFilters, ModelSortField } from '@workbench/models/library';
import type { ModelTaxonomyType } from '@workbench/models/types';

import { HStack, Icon, Input, InputGroup } from '@chakra-ui/react';
import { FilterMenuItem, ModelFilterMenu } from '@workbench/launchpad/models/shared/ModelFilterMenu';
import { SearchIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const SORT_FIELDS: { field: ModelSortField; labelKey: string }[] = [
  { field: 'default', labelKey: 'models.sort.default' },
  { field: 'name', labelKey: 'models.sort.name' },
  { field: 'base', labelKey: 'models.sort.base' },
  { field: 'size', labelKey: 'models.sort.size' },
  { field: 'format', labelKey: 'models.sort.format' },
];

const isFiltering = (filters: ModelLibraryFilters): boolean =>
  filters.typeFilter !== null || filters.baseFilter !== null || filters.missingOnly;

/**
 * Search box plus a combined filter/sort menu. The menu reflects only values
 * present in the current library so it never offers dead filters.
 */
export const ModelFilterBar = ({
  availableBases,
  availableTypes,
  filters,
  missingCount,
  onChange,
}: {
  availableBases: string[];
  availableTypes: ModelTaxonomyType[];
  filters: ModelLibraryFilters;
  missingCount: number;
  onChange: (filters: ModelLibraryFilters) => void;
}) => {
  const { t } = useTranslation();
  const handleSearchChange = useCallback(
    (searchTerm: string) => onChange({ ...filters, searchTerm }),
    [filters, onChange]
  );
  const handleMissingFilterToggle = useCallback(
    () => onChange({ ...filters, missingOnly: !filters.missingOnly, typeFilter: null }),
    [filters, onChange]
  );
  const handleBaseFilterChange = useCallback(
    (baseFilter: string | null) => onChange({ ...filters, baseFilter }),
    [filters, onChange]
  );
  const handleSortChange = useCallback(
    (sortField: ModelSortField, sortDirection: 'asc' | 'desc') => onChange({ ...filters, sortDirection, sortField }),
    [filters, onChange]
  );
  const handleTypeFilterChange = useCallback(
    (typeFilter: ModelTaxonomyType | null) => onChange({ ...filters, missingOnly: false, typeFilter }),
    [filters, onChange]
  );
  const extraTypeItems = useMemo(
    () =>
      missingCount > 0 ? (
        <FilterMenuItem
          isChecked={filters.missingOnly}
          label={t('models.missingFilesCount', { count: missingCount })}
          value="type-missing"
          onSelect={handleMissingFilterToggle}
        />
      ) : null,
    [filters.missingOnly, handleMissingFilterToggle, missingCount, t]
  );
  return (
    <HStack gap="1.5" w="full" px="3" pt="3">
      <InputGroup startElement={<Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />}>
        <Input
          aria-label={t('models.searchModels')}
          placeholder={t('models.searchModelsPlaceholder')}
          size="xs"
          value={filters.searchTerm}
          onChange={(event) => handleSearchChange(event.currentTarget.value)}
        />
      </InputGroup>
      <ModelFilterMenu
        ariaLabel={t('models.filterAndSort')}
        availableBases={availableBases}
        availableTypes={availableTypes}
        baseFilter={filters.baseFilter}
        extraTypeItems={extraTypeItems}
        isActive={isFiltering(filters)}
        sortDirection={filters.sortDirection}
        sortField={filters.sortField}
        sortFields={SORT_FIELDS}
        typeAllChecked={filters.typeFilter === null && !filters.missingOnly}
        typeFilter={filters.typeFilter}
        onBaseFilterChange={handleBaseFilterChange}
        onSortChange={handleSortChange}
        onTypeFilterChange={handleTypeFilterChange}
      />
    </HStack>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
