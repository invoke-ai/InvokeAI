/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { DEFAULT_NODE_PACK_FILTERS, type NodePackFilters, type NodePackSortField } from '@features/nodes/core/library';
import { IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { CheckIcon, SlidersHorizontalIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const SORT_FIELDS: readonly { field: NodePackSortField; labelKey: string }[] = [
  { field: 'name', labelKey: 'nodes.sort.name' },
  { field: 'nodeCount', labelKey: 'nodes.sort.nodeCount' },
  { field: 'path', labelKey: 'nodes.sort.path' },
];

/** Filter + sort menu for the pack library — the models filter menu's shape, scoped to pack axes. */
export const NodePackFilterMenu = ({
  filters,
  onChange,
}: {
  filters: NodePackFilters;
  onChange: (next: NodePackFilters) => void;
}) => {
  const { t } = useTranslation();
  const isActive =
    filters.problemsOnly ||
    filters.sortField !== DEFAULT_NODE_PACK_FILTERS.sortField ||
    filters.sortDirection !== DEFAULT_NODE_PACK_FILTERS.sortDirection;

  // Reselecting the active field flips direction (models menu behavior).
  const reportSort = (field: NodePackSortField) => {
    if (filters.sortField === field) {
      onChange({ ...filters, sortDirection: filters.sortDirection === 'asc' ? 'desc' : 'asc' });
    } else {
      onChange({ ...filters, sortDirection: 'asc', sortField: field });
    }
  };

  return (
    <Menu.Root closeOnSelect={false} positioning={{ placement: 'bottom-end' }}>
      <Menu.Trigger asChild>
        <IconButton
          aria-label={t('nodes.filterAndSort')}
          color={isActive ? 'accent.solid' : 'fg.muted'}
          size="xs"
          variant="outline"
        >
          <Icon as={SlidersHorizontalIcon} boxSize="4" />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="13rem">
            <Menu.ItemGroup>
              <FilterItem
                isChecked={filters.problemsOnly}
                label={t('nodes.problemPacksOnly')}
                value="problems-only"
                onSelect={() => onChange({ ...filters, problemsOnly: !filters.problemsOnly })}
              />
            </Menu.ItemGroup>
            <Menu.Separator />
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                {t('nodes.sortBy')}
              </Menu.ItemGroupLabel>
              {SORT_FIELDS.map(({ field, labelKey }) => (
                <FilterItem
                  key={field}
                  isChecked={filters.sortField === field}
                  label={t(labelKey)}
                  trailing={filters.sortField === field ? t(`common.${filters.sortDirection}`) : undefined}
                  value={field}
                  onSelect={() => reportSort(field)}
                />
              ))}
            </Menu.ItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const FilterItem = ({
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
