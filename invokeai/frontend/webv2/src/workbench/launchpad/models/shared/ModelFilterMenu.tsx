import type { ModelSortField } from '@workbench/models/library';
import type { ModelTaxonomyType } from '@workbench/models/types';
import type { ReactNode } from 'react';

import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { IconButton, MenuContent } from '@workbench/components/ui';
import { getModelBaseLabel } from '@workbench/models/baseIdentity';
import { getModelTypeLabel } from '@workbench/models/taxonomy';
import { CheckIcon, SlidersHorizontalIcon } from 'lucide-react';

export interface ModelFilterSortOption {
  field: ModelSortField;
  label: string;
}

const DEFAULT_SORT_FIELDS: ModelFilterSortOption[] = [
  { field: 'default', label: 'Default' },
  { field: 'name', label: 'Name' },
  { field: 'base', label: 'Base' },
  { field: 'size', label: 'Size' },
  { field: 'format', label: 'Format' },
];

/** Shared taxonomy filter + sort menu for installed and starter model lists. */
export const ModelFilterMenu = ({
  ariaLabel,
  availableBases,
  availableTypes,
  baseFilter,
  extraTypeItems,
  isActive,
  onBaseFilterChange,
  onSortChange,
  onTypeFilterChange,
  sortDirection,
  sortField,
  sortFields = DEFAULT_SORT_FIELDS,
  typeAllChecked,
  typeAllLabel = 'All models',
  typeFilter,
}: {
  ariaLabel: string;
  availableBases: string[];
  availableTypes: ModelTaxonomyType[];
  baseFilter: string | null;
  extraTypeItems?: ReactNode;
  isActive: boolean;
  onBaseFilterChange: (base: string | null) => void;
  onSortChange: (field: ModelSortField, direction: 'asc' | 'desc') => void;
  onTypeFilterChange: (type: ModelTaxonomyType | null) => void;
  sortDirection: 'asc' | 'desc';
  sortField: ModelSortField;
  sortFields?: readonly ModelFilterSortOption[];
  typeAllChecked?: boolean;
  typeAllLabel?: string;
  typeFilter: ModelTaxonomyType | null;
}) => (
  <Menu.Root closeOnSelect={false} positioning={{ placement: 'bottom-end' }}>
    <Menu.Trigger asChild>
      <IconButton aria-label={ariaLabel} color={isActive ? 'accent.solid' : 'fg.muted'} size="xs" variant="outline">
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
              isChecked={typeAllChecked ?? typeFilter === null}
              label={typeAllLabel}
              value="type-all"
              onSelect={() => onTypeFilterChange(null)}
            />
            {extraTypeItems}
            {availableTypes.map((type) => (
              <FilterMenuItem
                key={type}
                isChecked={typeFilter === type}
                label={getModelTypeLabel(type)}
                value={`type-${type}`}
                onSelect={() => onTypeFilterChange(typeFilter === type ? null : type)}
              />
            ))}
          </Menu.ItemGroup>
          <Menu.Separator />
          <Menu.ItemGroup>
            <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
              Base Architecture
            </Menu.ItemGroupLabel>
            <FilterMenuItem
              isChecked={baseFilter === null}
              label="All bases"
              value="base-all"
              onSelect={() => onBaseFilterChange(null)}
            />
            {availableBases.map((base) => (
              <FilterMenuItem
                key={base}
                isChecked={baseFilter === base}
                label={getModelBaseLabel(base)}
                value={`base-${base}`}
                onSelect={() => onBaseFilterChange(baseFilter === base ? null : base)}
              />
            ))}
          </Menu.ItemGroup>
          <Menu.Separator />
          <Menu.ItemGroup>
            <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
              Sort By
            </Menu.ItemGroupLabel>
            {sortFields.map(({ field, label }) => (
              <FilterMenuItem
                key={field}
                isChecked={sortField === field}
                label={label}
                value={`sort-${field}`}
                onSelect={() =>
                  onSortChange(field, sortField === field ? (sortDirection === 'asc' ? 'desc' : 'asc') : 'asc')
                }
                trailing={sortField === field ? (sortDirection === 'asc' ? 'Asc' : 'Desc') : undefined}
              />
            ))}
          </Menu.ItemGroup>
        </MenuContent>
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
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
