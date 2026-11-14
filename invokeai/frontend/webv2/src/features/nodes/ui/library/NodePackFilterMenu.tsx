import type { NodePackFilters, NodePackSortField } from '@features/nodes/core/library';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { ManagerFilterButton, ManagerFilterResetItem } from '@platform/ui/ManagerLayout';
import { MenuContent } from '@platform/ui/Menu';
import { CheckIcon } from 'lucide-react';
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
        <ManagerFilterButton activeCount={Number(filters.problemsOnly)} label={t('nodes.filterAndSort')} />
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
              <Menu.ItemGroupLabel color="fg.subtle" fontSize="xs" textTransform="uppercase">
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
            <ManagerFilterResetItem
              activeCount={Number(filters.problemsOnly)}
              onReset={() => onChange({ ...filters, problemsOnly: false })}
            />
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
    <Icon as={CheckIcon} boxSize="3" color="fg" opacity={isChecked ? 1 : 0} />
    <Menu.ItemText>{label}</Menu.ItemText>
    {trailing ? (
      <Text color="fg.subtle" fontSize="xs" ms="auto">
        {trailing}
      </Text>
    ) : null}
  </Menu.Item>
);
