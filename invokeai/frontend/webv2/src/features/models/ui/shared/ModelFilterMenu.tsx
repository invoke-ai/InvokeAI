/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelSortField } from '@features/models/core/library';
import type { ModelTaxonomyType } from '@features/models/core/types';
import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { getModelBaseLabel } from '@features/models/core/baseIdentity';
import { getModelTypeLabel } from '@features/models/core/taxonomy';
import { IconButton, MenuContent, Scrollable } from '@platform/ui';
import {
  ArrowUpDownIcon,
  CheckIcon,
  ChevronRightIcon,
  LayersIcon,
  ShapesIcon,
  SlidersHorizontalIcon,
} from 'lucide-react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const ROOT_POSITIONING = { placement: 'bottom-end' } as const;
const SUBMENU_POSITIONING = { placement: 'right-start' } as const;

export interface ModelFilterSortOption<Field extends ModelSortField = ModelSortField> {
  field: Field;
  labelKey: string;
}

/** The one source of sort options; derive narrower menus by filtering. */
export const SORT_FIELD_OPTIONS: readonly ModelFilterSortOption[] = [
  { field: 'default', labelKey: 'models.sort.default' },
  { field: 'name', labelKey: 'models.sort.name' },
  { field: 'base', labelKey: 'models.sort.base' },
  { field: 'type', labelKey: 'models.sort.type' },
  { field: 'size', labelKey: 'models.sort.size' },
  { field: 'format', labelKey: 'models.sort.format' },
  { field: 'path', labelKey: 'models.sort.path' },
];

/**
 * Shared taxonomy filter + sort menu for installed and starter model lists.
 * Generic over the sort-field subset so a menu fed a narrowed `sortFields`
 * list reports only those fields to `onSortChange`.
 */
export const ModelFilterMenu = <Field extends ModelSortField>({
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
  sortFields,
  typeAllChecked,
  typeFilter,
  typeSummary,
}: {
  ariaLabel: string;
  availableBases: string[];
  availableTypes: ModelTaxonomyType[];
  baseFilter: string | null;
  extraTypeItems?: ReactNode;
  isActive: boolean;
  onBaseFilterChange: (base: string | null) => void;
  onSortChange: (field: Field, direction: 'asc' | 'desc') => void;
  onTypeFilterChange: (type: ModelTaxonomyType | null) => void;
  sortDirection: 'asc' | 'desc';
  sortField: Field;
  sortFields: readonly ModelFilterSortOption<Field>[];
  typeAllChecked?: boolean;
  typeFilter: ModelTaxonomyType | null;
  /**
   * What the Model Type row should read when the filter is neither "all" nor
   * one of `availableTypes` — the caller's own pseudo-type from
   * `extraTypeItems`, which this component cannot name for itself.
   */
  typeSummary?: string;
}) => {
  const { t } = useTranslation();
  // Safe to widen: the menu only reports fields drawn from `sortFields`.
  const reportSort = onSortChange as (field: ModelSortField, direction: 'asc' | 'desc') => void;
  const isAllTypes = typeAllChecked ?? typeFilter === null;
  const sortLabelKey = sortFields.find((option) => option.field === sortField)?.labelKey;
  const sortDirectionLabel = sortDirection === 'asc' ? t('common.asc') : t('common.desc');
  // `default` is the catalog's own order, so a direction on it would name
  // something the user cannot act on.
  const sortSummary = sortLabelKey
    ? sortField === 'default'
      ? t(sortLabelKey)
      : `${t(sortLabelKey)} · ${sortDirectionLabel}`
    : undefined;

  return (
    <Menu.Root closeOnSelect={false} positioning={ROOT_POSITIONING}>
      <Menu.Trigger asChild>
        <IconButton aria-label={ariaLabel} color={isActive ? 'accent.solid' : 'fg.muted'} size="xs" variant="outline">
          <Icon as={SlidersHorizontalIcon} boxSize="4" />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="15rem" py="1">
            <FilterSubMenu
              icon={ShapesIcon}
              label={t('models.modelType')}
              summary={typeFilter ? getModelTypeLabel(typeFilter) : isAllTypes ? t('models.allModels') : typeSummary}
            >
              <AllTypesFilterMenuItem
                isChecked={isAllTypes}
                label={t('models.allModels')}
                onTypeFilterChange={onTypeFilterChange}
              />
              {extraTypeItems}
              {availableTypes.map((type) => (
                <TypeFilterMenuItem
                  key={type}
                  type={type}
                  typeFilter={typeFilter}
                  onTypeFilterChange={onTypeFilterChange}
                />
              ))}
            </FilterSubMenu>
            <FilterSubMenu
              isScrollable
              icon={LayersIcon}
              label={t('models.baseArchitecture')}
              summary={baseFilter ? getModelBaseLabel(baseFilter) : t('models.allBases')}
            >
              <AllBasesFilterMenuItem isChecked={baseFilter === null} onBaseFilterChange={onBaseFilterChange} />
              {availableBases.map((base) => (
                <BaseFilterMenuItem
                  key={base}
                  base={base}
                  baseFilter={baseFilter}
                  onBaseFilterChange={onBaseFilterChange}
                />
              ))}
            </FilterSubMenu>
            <Menu.Separator />
            <FilterSubMenu icon={ArrowUpDownIcon} label={t('models.sortBy')} summary={sortSummary}>
              {sortFields.map(({ field, labelKey }) => (
                <SortFilterMenuItem
                  key={field}
                  field={field}
                  labelKey={labelKey}
                  sortDirection={sortDirection}
                  sortField={sortField}
                  onSortChange={reportSort}
                />
              ))}
            </FilterSubMenu>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/**
 * One filter dimension, folded away behind its own row.
 *
 * The three groups used to stack in a single panel that scrolled at 70vh — on
 * a library with a dozen base architectures, choosing a sort order meant
 * scrolling past every one of them. Folding each into a submenu costs a
 * hover, so the row carries its current value: the state stays readable
 * without opening anything, which is the only reason the flat list was worth
 * keeping.
 */
const FilterSubMenu = ({
  children,
  icon,
  isScrollable = false,
  label,
  summary,
}: {
  children: ReactNode;
  icon: LucideIcon;
  isScrollable?: boolean;
  label: string;
  summary?: string;
}) => (
  <Menu.Root closeOnSelect={false} positioning={SUBMENU_POSITIONING}>
    <Menu.TriggerItem>
      <HStack gap="2" minW="0" w="full">
        <Icon as={icon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
        <Text flexShrink={0} fontSize="xs">
          {label}
        </Text>
        {summary ? (
          <Text color="fg.subtle" fontSize="2xs" ms="auto" minW="0" truncate>
            {summary}
          </Text>
        ) : null}
        <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} ms={summary ? undefined : 'auto'} />
      </HStack>
    </Menu.TriggerItem>
    <Portal>
      <Menu.Positioner>
        <MenuContent minW="13rem" py="1">
          {isScrollable ? <Scrollable maxH="60vh">{children}</Scrollable> : children}
        </MenuContent>
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);

/** Checkmark-style menu item shared by the model filter menus. */
interface FilterMenuItemProps {
  isChecked: boolean;
  label: string;
  onSelect: () => void;
  trailing?: string;
  value: string;
}

export const FilterMenuItem = memo(function FilterMenuItem({
  isChecked,
  label,
  onSelect,
  trailing,
  value,
}: FilterMenuItemProps) {
  return (
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
});

const AllTypesFilterMenuItem = memo(function AllTypesFilterMenuItem({
  isChecked,
  label,
  onTypeFilterChange,
}: {
  isChecked: boolean;
  label: string;
  onTypeFilterChange: (type: ModelTaxonomyType | null) => void;
}) {
  return (
    <FilterMenuItem isChecked={isChecked} label={label} value="type-all" onSelect={() => onTypeFilterChange(null)} />
  );
});

const TypeFilterMenuItem = memo(function TypeFilterMenuItem({
  type,
  typeFilter,
  onTypeFilterChange,
}: {
  type: ModelTaxonomyType;
  typeFilter: ModelTaxonomyType | null;
  onTypeFilterChange: (type: ModelTaxonomyType | null) => void;
}) {
  return (
    <FilterMenuItem
      isChecked={typeFilter === type}
      label={getModelTypeLabel(type)}
      value={`type-${type}`}
      onSelect={() => onTypeFilterChange(typeFilter === type ? null : type)}
    />
  );
});

const AllBasesFilterMenuItem = memo(function AllBasesFilterMenuItem({
  isChecked,
  onBaseFilterChange,
}: {
  isChecked: boolean;
  onBaseFilterChange: (base: string | null) => void;
}) {
  const { t } = useTranslation();

  return (
    <FilterMenuItem
      isChecked={isChecked}
      label={t('models.allBases')}
      value="base-all"
      onSelect={() => onBaseFilterChange(null)}
    />
  );
});

const BaseFilterMenuItem = memo(function BaseFilterMenuItem({
  base,
  baseFilter,
  onBaseFilterChange,
}: {
  base: string;
  baseFilter: string | null;
  onBaseFilterChange: (base: string | null) => void;
}) {
  return (
    <FilterMenuItem
      isChecked={baseFilter === base}
      label={getModelBaseLabel(base)}
      value={`base-${base}`}
      onSelect={() => onBaseFilterChange(baseFilter === base ? null : base)}
    />
  );
});

const SortFilterMenuItem = memo(function SortFilterMenuItem({
  field,
  labelKey,
  sortDirection,
  sortField,
  onSortChange,
}: {
  field: ModelSortField;
  labelKey: string;
  sortDirection: 'asc' | 'desc';
  sortField: ModelSortField;
  onSortChange: (field: ModelSortField, direction: 'asc' | 'desc') => void;
}) {
  const { t } = useTranslation();

  return (
    <FilterMenuItem
      isChecked={sortField === field}
      label={t(labelKey)}
      value={`sort-${field}`}
      onSelect={() => onSortChange(field, sortField === field ? (sortDirection === 'asc' ? 'desc' : 'asc') : 'asc')}
      trailing={sortField === field ? (sortDirection === 'asc' ? t('common.asc') : t('common.desc')) : undefined}
    />
  );
});
