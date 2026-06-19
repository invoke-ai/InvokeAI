import type { ModelLibraryFilters, ModelSortField } from '@workbench/models/library';
import type { ModelTaxonomyType } from '@workbench/models/types';

import { HStack, Icon, Input, InputGroup } from '@chakra-ui/react';
import { FilterMenuItem, ModelFilterMenu } from '@workbench/launchpad/models/shared/ModelFilterMenu';
import { SearchIcon } from 'lucide-react';

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
}) => (
  <HStack gap="1.5" w="full" px="3" pt="3">
    <InputGroup startElement={<Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />}>
      <Input
        aria-label="Search models"
        placeholder="Search models…"
        size="xs"
        value={filters.searchTerm}
        onChange={(event) => onChange({ ...filters, searchTerm: event.currentTarget.value })}
      />
    </InputGroup>
    <ModelFilterMenu
      ariaLabel="Filter and sort models"
      availableBases={availableBases}
      availableTypes={availableTypes}
      baseFilter={filters.baseFilter}
      extraTypeItems={
        missingCount > 0 ? (
          <FilterMenuItem
            isChecked={filters.missingOnly}
            label={`Missing files (${missingCount})`}
            value="type-missing"
            onSelect={() => onChange({ ...filters, missingOnly: !filters.missingOnly, typeFilter: null })}
          />
        ) : null
      }
      isActive={isFiltering(filters)}
      sortDirection={filters.sortDirection}
      sortField={filters.sortField}
      sortFields={SORT_FIELDS}
      typeAllChecked={filters.typeFilter === null && !filters.missingOnly}
      typeFilter={filters.typeFilter}
      onBaseFilterChange={(baseFilter) => onChange({ ...filters, baseFilter })}
      onSortChange={(sortField, sortDirection) => onChange({ ...filters, sortDirection, sortField })}
      onTypeFilterChange={(typeFilter) => onChange({ ...filters, missingOnly: false, typeFilter })}
    />
  </HStack>
);
