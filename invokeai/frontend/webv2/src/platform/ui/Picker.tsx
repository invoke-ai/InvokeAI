import type { ReactNode } from 'react';

import { Box, HStack, Icon, Input, InputGroup, ScrollArea, Spacer, Stack, Text } from '@chakra-ui/react';
import { dropdownGroupLabel } from '@theme/recipes';
import { SearchIcon } from 'lucide-react';
import { useCallback, useDeferredValue, useMemo, useRef, useState } from 'react';

export interface PickerGroup<T> {
  id: string;
  name: string;
  options: T[];
  shortName?: string;
  colorPalette?: string;
  getCountLabel?: (count: number) => string;
}

export interface PickerOptionState {
  isActive: boolean;
  isCompact: boolean;
  isSelected: boolean;
}

const SEARCH_ICON = <Icon as={SearchIcon} size="xs" />;

const flattenGroups = <T,>(groups: PickerGroup<T>[]): T[] => groups.flatMap((group) => group.options);

export const Picker = <T,>({
  emptyMessage,
  getIsOptionDisabled,
  getOptionId,
  groups,
  isCompact = false,
  isMatch,
  listLabel,
  noMatchesMessage,
  renderOption,
  searchPlaceholder,
  searchSlot,
  selectedId,
  statusSlot,
  toolbarSlot,
  onSelect,
}: {
  emptyMessage: ReactNode;
  getIsOptionDisabled?: (option: T) => boolean;
  getOptionId: (option: T) => string;
  groups: PickerGroup<T>[];
  isCompact?: boolean;
  isMatch: (option: T, searchTerm: string) => boolean;
  listLabel: string;
  noMatchesMessage: ReactNode;
  renderOption: (option: T, state: PickerOptionState) => ReactNode;
  searchPlaceholder: string;
  searchSlot?: ReactNode;
  selectedId: string | null;
  statusSlot?: ReactNode;
  toolbarSlot?: ReactNode;
  onSelect: (option: T) => void;
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeId, setActiveId] = useState<string | null>(null);
  const deferredSearchTerm = useDeferredValue(searchTerm);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const visibleGroups = useMemo(() => {
    const term = deferredSearchTerm.trim();

    if (!term) {
      return groups.filter((group) => group.options.length > 0);
    }

    return groups
      .map((group) => ({ ...group, options: group.options.filter((option) => isMatch(option, term)) }))
      .filter((group) => group.options.length > 0);
  }, [deferredSearchTerm, groups, isMatch]);

  const flatOptions = useMemo(() => flattenGroups(visibleGroups), [visibleGroups]);
  const selectableIds = useMemo(
    () => flatOptions.filter((option) => !getIsOptionDisabled?.(option)).map(getOptionId),
    [flatOptions, getIsOptionDisabled, getOptionId]
  );

  // The active row must always exist in the current result set — a search that
  // filters it away silently hands the highlight to the first remaining row.
  const resolvedActiveId =
    activeId && selectableIds.includes(activeId)
      ? activeId
      : (selectableIds.find((id) => id === selectedId) ?? selectableIds[0] ?? null);

  const moveActive = useCallback(
    (delta: 1 | -1) => {
      if (selectableIds.length === 0) {
        return;
      }

      const currentIndex = resolvedActiveId ? selectableIds.indexOf(resolvedActiveId) : -1;
      const nextIndex = Math.min(Math.max(currentIndex + delta, 0), selectableIds.length - 1);
      const nextId = selectableIds[nextIndex] ?? null;

      setActiveId(nextId);

      if (nextId) {
        listRef.current
          ?.querySelector(`[data-picker-option-id="${CSS.escape(nextId)}"]`)
          ?.scrollIntoView({ block: 'nearest' });
      }
    },
    [resolvedActiveId, selectableIds]
  );

  const handleSearchKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      // The workbench hotkey runtime listens on the window; typing here is not
      // a command sequence.
      event.stopPropagation();

      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        moveActive(event.key === 'ArrowDown' ? 1 : -1);
        return;
      }

      if (event.key === 'Enter' && resolvedActiveId) {
        const option = flatOptions.find((candidate) => getOptionId(candidate) === resolvedActiveId);

        if (option) {
          event.preventDefault();
          onSelect(option);
        }
      }
    },
    [flatOptions, getOptionId, moveActive, onSelect, resolvedActiveId]
  );

  const handleSearchChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.currentTarget.value);
    setActiveId(null);
  }, []);

  const hasAnyOption = groups.some((group) => group.options.length > 0);

  return (
    <Stack gap="0" minH="0">
      <Stack gap="2" p="2">
        <HStack gap="1">
          <InputGroup flex="1" minW="0" startElement={SEARCH_ICON}>
            <Input
              ref={searchInputRef}
              aria-activedescendant={resolvedActiveId ? `picker-option-${resolvedActiveId}` : undefined}
              aria-controls={listLabel}
              aria-label={searchPlaceholder}
              autoComplete="off"
              placeholder={searchPlaceholder}
              size="xs"
              value={searchTerm}
              onChange={handleSearchChange}
              onKeyDown={handleSearchKeyDown}
            />
          </InputGroup>
          {searchSlot}
        </HStack>
        {toolbarSlot}
      </Stack>
      <Box borderColor="border.subtle" borderTopWidth="1px" />
      <ScrollArea.Root maxH="18rem" size="xs" variant="hover" w="full">
        <ScrollArea.Viewport ref={listRef} maxH="inherit" w="full">
          <ScrollArea.Content aria-label={listLabel} maxW="full" minW="0" py="1" role="listbox" w="full">
            {statusSlot ??
              (!hasAnyOption ? (
                <Text color="fg.subtle" fontSize="2xs" p="2">
                  {emptyMessage}
                </Text>
              ) : visibleGroups.length === 0 ? (
                <Text color="fg.subtle" fontSize="2xs" p="2">
                  {noMatchesMessage}
                </Text>
              ) : (
                visibleGroups.map((group) => (
                  <PickerGroupSection
                    key={group.id}
                    activeId={resolvedActiveId}
                    getIsOptionDisabled={getIsOptionDisabled}
                    getOptionId={getOptionId}
                    group={group}
                    isCompact={isCompact}
                    renderOption={renderOption}
                    selectedId={selectedId}
                    showHeader={groups.length > 1}
                    onActivate={setActiveId}
                    onSelect={onSelect}
                  />
                ))
              ))}
          </ScrollArea.Content>
        </ScrollArea.Viewport>
        {/*
          Above the sticky group headers. They paint an opaque background at
          z-index 1, which would otherwise chop the scrollbar into one segment
          per group.
        */}
        <ScrollArea.Scrollbar zIndex="2">
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>
    </Stack>
  );
};

const PickerGroupSection = <T,>({
  activeId,
  getIsOptionDisabled,
  getOptionId,
  group,
  isCompact,
  renderOption,
  selectedId,
  showHeader,
  onActivate,
  onSelect,
}: {
  activeId: string | null;
  getIsOptionDisabled?: (option: T) => boolean;
  getOptionId: (option: T) => string;
  group: PickerGroup<T>;
  isCompact: boolean;
  renderOption: (option: T, state: PickerOptionState) => ReactNode;
  selectedId: string | null;
  showHeader: boolean;
  onActivate: (id: string) => void;
  onSelect: (option: T) => void;
}) => {
  const railColor = group.colorPalette ? `${group.colorPalette}.solid` : 'border.emphasized';
  const countLabel = group.getCountLabel?.(group.options.length);

  // `w`/`minW` are load-bearing: ScrollArea's content wrapper sizes to
  // max-content, so without them a long model name widens the whole list and
  // every group grows a horizontal scrollbar instead of truncating.
  return (
    <Box
      borderInlineStartColor={showHeader ? railColor : undefined}
      borderInlineStartWidth={showHeader ? '2px' : '0'}
      minW="0"
      w="full"
    >
      {showHeader ? (
        <HStack bg="bg.panel" gap="2" minW="0" pe="3" position="sticky" ps="2" py="1" top="0" w="full" zIndex="1">
          <Text color={railColor} css={dropdownGroupLabel} minW="0" truncate>
            {group.name}
          </Text>
          <Spacer />
          {countLabel ? (
            <Text color="fg.subtle" flexShrink={0} fontSize="2xs">
              {countLabel}
            </Text>
          ) : null}
        </HStack>
      ) : null}
      {group.options.map((option) => {
        const id = getOptionId(option);
        const isDisabled = getIsOptionDisabled?.(option) ?? false;

        return (
          <PickerOptionRow
            key={id}
            id={id}
            isActive={activeId === id}
            isCompact={isCompact}
            isDisabled={isDisabled}
            isSelected={selectedId === id}
            option={option}
            renderOption={renderOption}
            onActivate={onActivate}
            onSelect={onSelect}
          />
        );
      })}
    </Box>
  );
};

const PickerOptionRow = <T,>({
  id,
  isActive,
  isCompact,
  isDisabled,
  isSelected,
  option,
  renderOption,
  onActivate,
  onSelect,
}: {
  id: string;
  isActive: boolean;
  isCompact: boolean;
  isDisabled: boolean;
  isSelected: boolean;
  option: T;
  renderOption: (option: T, state: PickerOptionState) => ReactNode;
  onActivate: (id: string) => void;
  onSelect: (option: T) => void;
}) => {
  const handleClick = useCallback(() => {
    if (!isDisabled) {
      onSelect(option);
    }
  }, [isDisabled, onSelect, option]);
  const handlePointerMove = useCallback(() => {
    if (!isDisabled) {
      onActivate(id);
    }
  }, [id, isDisabled, onActivate]);

  return (
    <Box
      aria-disabled={isDisabled || undefined}
      aria-selected={isSelected}
      bg={isActive && !isDisabled ? 'bg.emphasized' : undefined}
      cursor={isDisabled ? 'not-allowed' : 'pointer'}
      data-active={isActive ? '' : undefined}
      data-picker-option-id={id}
      id={`picker-option-${id}`}
      minW="0"
      opacity={isDisabled ? 0.4 : undefined}
      px="2"
      py={isCompact ? '0.5' : '1'}
      role="option"
      tabIndex={-1}
      w="full"
      onClick={handleClick}
      onPointerMove={handlePointerMove}
    >
      {renderOption(option, { isActive, isCompact, isSelected })}
    </Box>
  );
};
