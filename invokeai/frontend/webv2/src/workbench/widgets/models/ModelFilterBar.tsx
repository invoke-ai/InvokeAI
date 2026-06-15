import { HStack, Icon, Input, InputGroup, Menu, Portal, Text } from '@chakra-ui/react';
import { CheckIcon, SearchIcon, SlidersHorizontalIcon } from 'lucide-react';

import { IconButton } from '../../components/ui/Button';
import { MenuContent } from '../../components/ui/Menu';
import type { ModelLibraryFilters, ModelSortField } from '../../models/library';
import { getModelBaseLabel, getModelTypeLabel } from '../../models/taxonomy';
import type { ModelTaxonomyType } from '../../models/types';

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
  <HStack gap="1.5" w="full">
    <InputGroup startElement={<Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />}>
      <Input
        aria-label="Search models"
        placeholder="Search models…"
        size="sm"
        value={filters.searchTerm}
        onChange={(event) => onChange({ ...filters, searchTerm: event.currentTarget.value })}
      />
    </InputGroup>
    <Menu.Root closeOnSelect={false} positioning={{ placement: 'bottom-end' }}>
      <Menu.Trigger asChild>
        <IconButton
          aria-label="Filter and sort models"
          color={isFiltering(filters) ? 'accent.solid' : 'fg.muted'}
          size="sm"
          variant="ghost"
        >
          <Icon as={SlidersHorizontalIcon} boxSize="4" />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent maxH="70vh" minW="13rem" overflowY="auto" py="1">
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                Model Type
              </Menu.ItemGroupLabel>
              <FilterMenuItem
                isChecked={filters.typeFilter === null && !filters.missingOnly}
                label="All models"
                value="type-all"
                onSelect={() => onChange({ ...filters, missingOnly: false, typeFilter: null })}
              />
              {missingCount > 0 ? (
                <FilterMenuItem
                  isChecked={filters.missingOnly}
                  label={`Missing files (${missingCount})`}
                  value="type-missing"
                  onSelect={() => onChange({ ...filters, missingOnly: !filters.missingOnly, typeFilter: null })}
                />
              ) : null}
              {availableTypes.map((type) => (
                <FilterMenuItem
                  key={type}
                  isChecked={filters.typeFilter === type}
                  label={getModelTypeLabel(type)}
                  value={`type-${type}`}
                  onSelect={() =>
                    onChange({
                      ...filters,
                      missingOnly: false,
                      typeFilter: filters.typeFilter === type ? null : type,
                    })
                  }
                />
              ))}
            </Menu.ItemGroup>
            <Menu.Separator />
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                Base Architecture
              </Menu.ItemGroupLabel>
              <FilterMenuItem
                isChecked={filters.baseFilter === null}
                label="All bases"
                value="base-all"
                onSelect={() => onChange({ ...filters, baseFilter: null })}
              />
              {availableBases.map((base) => (
                <FilterMenuItem
                  key={base}
                  isChecked={filters.baseFilter === base}
                  label={getModelBaseLabel(base)}
                  value={`base-${base}`}
                  onSelect={() => onChange({ ...filters, baseFilter: filters.baseFilter === base ? null : base })}
                />
              ))}
            </Menu.ItemGroup>
            <Menu.Separator />
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                Sort By
              </Menu.ItemGroupLabel>
              {SORT_FIELDS.map(({ field, label }) => (
                <FilterMenuItem
                  key={field}
                  isChecked={filters.sortField === field}
                  label={label}
                  value={`sort-${field}`}
                  onSelect={() =>
                    onChange(
                      filters.sortField === field
                        ? { ...filters, sortDirection: filters.sortDirection === 'asc' ? 'desc' : 'asc' }
                        : { ...filters, sortDirection: 'asc', sortField: field }
                    )
                  }
                  trailing={
                    filters.sortField === field ? (filters.sortDirection === 'asc' ? 'Asc' : 'Desc') : undefined
                  }
                />
              ))}
            </Menu.ItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  </HStack>
);

/** Checkmark-style menu item shared by the model filter menus. */
export const FilterMenuItem = ({
  isChecked,
  label,
  onSelect,
  trailing,
  value,
}: {
  isChecked: boolean;
  label: string;
  onSelect: () => void;
  trailing?: string;
  value: string;
}) => (
  <Menu.Item aria-checked={isChecked} closeOnSelect={false} role="menuitemcheckbox" value={value} onClick={onSelect}>
    <Icon as={CheckIcon} boxSize="3" opacity={isChecked ? 1 : 0} />
    <Menu.ItemText fontSize="xs">{label}</Menu.ItemText>
    {trailing ? (
      <Text color="fg.subtle" fontSize="2xs" ms="auto">
        {trailing}
      </Text>
    ) : null}
  </Menu.Item>
);
