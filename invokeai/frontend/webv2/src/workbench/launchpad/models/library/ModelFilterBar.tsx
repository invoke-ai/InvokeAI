/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelLibraryFilters, ModelSortField } from '@workbench/models/library';
import type { ModelTaxonomyType } from '@workbench/models/types';

import { HStack, Icon, Input, InputGroup } from '@chakra-ui/react';
import { FilterMenuItem, ModelFilterMenu } from '@workbench/launchpad/models/shared/ModelFilterMenu';
import { SearchIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';

const SORT_FIELDS: { field: ModelSortField; label: string }[] = [
  { field: 'default', label: 'Default' },
  { field: 'name', label: 'Name' },
  { field: 'base', label: 'Base' },
  { field: 'size', label: 'Size' },
  { field: 'format', label: 'Format' },
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
          label={`Missing files (${missingCount})`}
          value="type-missing"
          onSelect={handleMissingFilterToggle}
        />
      ) : null,
    [filters.missingOnly, handleMissingFilterToggle, missingCount]
  );

  return (
    <HStack gap="1.5" w="full" px="3" pt="3">
      <InputGroup startElement={<Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />}>
        <Input
          aria-label="Search models"
          placeholder="Search models…"
          size="xs"
          value={filters.searchTerm}
          onChange={(event) => handleSearchChange(event.currentTarget.value)}
        />
      </InputGroup>
      <ModelFilterMenu
        ariaLabel="Filter and sort models"
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
