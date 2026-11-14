import type { WorkflowLibraryLayout, WorkflowThumbnailFit } from '@features/workflow/ui/WorkflowUiContext';
import type { ChangeEvent, ReactNode, RefObject } from 'react';

import { Box, HStack, Icon, Input, InputGroup, Menu, Portal, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import {
  ArrowDownWideNarrowIcon,
  ArrowUpNarrowWideIcon,
  ChevronDownIcon,
  LayoutGridIcon,
  ListIcon,
  PlusIcon,
  SearchIcon,
  SlidersHorizontalIcon,
} from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowLibraryCardSize } from './workflowThumbnail';

import { WORKFLOW_LIBRARY_NARROW, WORKFLOW_LIBRARY_WIDE_ONLY_CSS } from './WorkflowLibraryRail';

/**
 * The library's one toolbar, the same on every place: one row of equal-height controls 8px apart. Wide: search, the
 * Sort menu (key and direction, its glyph showing the direction), the view menu (grid or list, its glyph showing which,
 * and for the grid card size and thumbnail fit) and, on This project, New workflow at the row's end. Narrow
 * (`WORKFLOW_LIBRARY_NARROW`): the places select and Tags button lead, and one menu holds the order and view choices.
 */

const SEARCH_START_ELEMENT = <Icon aria-hidden as={SearchIcon} boxSize="3.5" color="fg.subtle" />;
const MENU_POSITIONING = { placement: 'bottom-end' } as const;
const NARROW_ONLY_CSS = { display: 'none', [WORKFLOW_LIBRARY_NARROW]: { display: 'inline-flex' } } as const;
const NARROW_LEAD_CSS = {
  display: 'none',
  [WORKFLOW_LIBRARY_NARROW]: { alignItems: 'center', display: 'flex', gap: '2' },
} as const;
// The key's natural direction first: dates newest first, names A to Z, the project's own order first to last.
const DESCENDING_FIRST = ['desc', 'asc'] as const;
const ASCENDING_FIRST = ['asc', 'desc'] as const;
const CARD_SIZES = ['small', 'large'] as const;
const LAYOUTS = ['grid', 'list'] as const;

export interface WorkflowLibrarySortOption {
  label: string;
  value: string;
}

export interface WorkflowLibraryToolbarProps {
  /** The narrow layout's places select and Tags button, leading the row; the wide layout shows them in the rail. */
  narrowLead?: ReactNode;
  search: string;
  /** The dialog's initial focus. */
  searchInputRef: RefObject<HTMLInputElement | null>;
  sortOptions: readonly WorkflowLibrarySortOption[];
  sort: string;
  /** A place with one fixed order shows it without offering a choice. */
  isSortFixed: boolean;
  direction: 'asc' | 'desc';
  /** What each direction means for the current key, such as "Newest first". */
  directionLabels: Readonly<Record<'asc' | 'desc', string>>;
  isDirectionFixed: boolean;
  /** Grid of cards or list rows; card size and thumbnail fit apply to the grid only. */
  layout: WorkflowLibraryLayout;
  cardSize: WorkflowLibraryCardSize;
  thumbnailFit: WorkflowThumbnailFit;
  /** Offered on This project only. */
  onNewWorkflow?: () => void;
  onSearchChange: (search: string) => void;
  onSortChange: (sort: string) => void;
  onDirectionToggle: () => void;
  onLayoutChange: (layout: WorkflowLibraryLayout) => void;
  onCardSizeChange: (cardSize: WorkflowLibraryCardSize) => void;
  onThumbnailFitChange: (fit: WorkflowThumbnailFit) => void;
}

export const WorkflowLibraryToolbar = ({
  cardSize,
  direction,
  directionLabels,
  isDirectionFixed,
  isSortFixed,
  layout,
  narrowLead,
  search,
  searchInputRef,
  sort,
  sortOptions,
  thumbnailFit,
  onCardSizeChange,
  onDirectionToggle,
  onLayoutChange,
  onNewWorkflow,
  onSearchChange,
  onSortChange,
  onThumbnailFitChange,
}: WorkflowLibraryToolbarProps) => {
  const { t } = useTranslation();
  const viewIds = useTooltipTriggerIds();
  const narrowOptionsIds = useTooltipTriggerIds();
  const sortLabel = sortOptions.find((option) => option.value === sort)?.label ?? '';
  const sortName = t('workflowLibrary.sortControl', { direction: directionLabels[direction], sort: sortLabel });

  const handleSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => onSearchChange(event.currentTarget.value),
    [onSearchChange]
  );
  const handleSortChange = useCallback(({ value }: { value: string }) => onSortChange(value), [onSortChange]);
  const handleDirectionChange = useCallback(
    ({ value }: { value: string }) => {
      if (value !== direction) {
        onDirectionToggle();
      }
    },
    [direction, onDirectionToggle]
  );
  const handleLayoutChange = useCallback(
    ({ value }: { value: string }) => onLayoutChange(value === 'list' ? 'list' : 'grid'),
    [onLayoutChange]
  );
  const handleCardSizeChange = useCallback(
    ({ value }: { value: string }) => onCardSizeChange(value === 'large' ? 'large' : 'small'),
    [onCardSizeChange]
  );
  const handleFitChange = useCallback(
    ({ value }: { value: string }) => onThumbnailFitChange(value === 'contain' ? 'contain' : 'cover'),
    [onThumbnailFitChange]
  );

  // The order: the key, then its two directions spelled out for that key (Newest first / Oldest first, A to Z / Z to A).
  const orderGroups = (
    <>
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('workflowLibrary.sortLabel')}</Menu.ItemGroupLabel>
        <Menu.RadioItemGroup value={sort} onValueChange={handleSortChange}>
          {sortOptions.map((option) => (
            <Menu.RadioItem key={option.value} disabled={isSortFixed} value={option.value}>
              <Menu.ItemIndicator />
              <Menu.ItemText>{option.label}</Menu.ItemText>
            </Menu.RadioItem>
          ))}
        </Menu.RadioItemGroup>
      </Menu.ItemGroup>
      <Menu.Separator />
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('workflowLibrary.sortDirectionGroup')}</Menu.ItemGroupLabel>
        <Menu.RadioItemGroup value={direction} onValueChange={handleDirectionChange}>
          {(sort === 'name' || sort === 'order' ? ASCENDING_FIRST : DESCENDING_FIRST).map((value) => (
            <Menu.RadioItem key={value} disabled={isDirectionFixed} value={value}>
              <Menu.ItemIndicator />
              <Menu.ItemText>{directionLabels[value]}</Menu.ItemText>
            </Menu.RadioItem>
          ))}
        </Menu.RadioItemGroup>
      </Menu.ItemGroup>
    </>
  );
  // Card size and thumbnail display shape only the grid; the list shows neither.
  const gridViewGroups = (
    <>
      <Menu.Separator />
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('workflowLibrary.cardSize')}</Menu.ItemGroupLabel>
        <Menu.RadioItemGroup value={cardSize} onValueChange={handleCardSizeChange}>
          {CARD_SIZES.map((value) => (
            <Menu.RadioItem key={value} value={value}>
              <Menu.ItemIndicator />
              <Menu.ItemText>
                {t(value === 'large' ? 'workflowLibrary.cardSizeLarge' : 'workflowLibrary.cardSizeSmall')}
              </Menu.ItemText>
            </Menu.RadioItem>
          ))}
        </Menu.RadioItemGroup>
      </Menu.ItemGroup>
      <Menu.Separator />
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('workflowLibrary.thumbnailFit')}</Menu.ItemGroupLabel>
        <Menu.RadioItemGroup value={thumbnailFit} onValueChange={handleFitChange}>
          <Menu.RadioItem value="cover">
            <Menu.ItemIndicator />
            <Menu.ItemText>{t('workflowLibrary.thumbnailFitCover')}</Menu.ItemText>
          </Menu.RadioItem>
          <Menu.RadioItem value="contain">
            <Menu.ItemIndicator />
            <Menu.ItemText>{t('workflowLibrary.thumbnailFitContain')}</Menu.ItemText>
          </Menu.RadioItem>
        </Menu.RadioItemGroup>
      </Menu.ItemGroup>
    </>
  );
  const viewGroups = (
    <>
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('workflowLibrary.layoutGroup')}</Menu.ItemGroupLabel>
        <Menu.RadioItemGroup value={layout} onValueChange={handleLayoutChange}>
          {LAYOUTS.map((value) => (
            <Menu.RadioItem key={value} value={value}>
              <Menu.ItemIndicator />
              <Menu.ItemText>
                {t(value === 'list' ? 'workflowLibrary.layoutList' : 'workflowLibrary.layoutGrid')}
              </Menu.ItemText>
            </Menu.RadioItem>
          ))}
        </Menu.RadioItemGroup>
      </Menu.ItemGroup>
      {layout === 'grid' ? gridViewGroups : null}
    </>
  );

  return (
    // One row; only a very narrow dialog wraps the menus under the search.
    <HStack data-workflow-library-toolbar flexShrink={0} flexWrap="wrap" gap="2" minW="0">
      {narrowLead ? (
        <Box css={NARROW_LEAD_CSS} flexShrink={0} maxW="full" minW="0">
          {narrowLead}
        </Box>
      ) : null}
      <InputGroup flex="1" minW="8rem" startElement={SEARCH_START_ELEMENT}>
        <Input
          ref={searchInputRef}
          aria-label={t('workflowLibrary.searchPlaceholder')}
          placeholder={t('workflowLibrary.searchPlaceholder')}
          type="search"
          value={search}
          onChange={handleSearchChange}
        />
      </InputGroup>
      {/* Its text names the key; the glyph shows the direction, and the accessible name spells out both. */}
      <Box css={WORKFLOW_LIBRARY_WIDE_ONLY_CSS} flexShrink={0}>
        <Menu.Root closeOnSelect={false} positioning={MENU_POSITIONING}>
          <Menu.Trigger asChild>
            <Button
              aria-label={sortName}
              data-workflow-library-sort
              disabled={isSortFixed && isDirectionFixed}
              gap="1.5"
              variant="outline"
            >
              <Icon
                aria-hidden
                as={direction === 'asc' ? ArrowUpNarrowWideIcon : ArrowDownWideNarrowIcon}
                boxSize="3.5"
                color="fg.muted"
              />
              <Text as="span" whiteSpace="nowrap">
                {sortLabel}
              </Text>
              <Icon aria-hidden as={ChevronDownIcon} boxSize="3.5" color="fg.muted" />
            </Button>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent maxH="var(--available-height)" minW="13rem" overflowY="auto">
                {orderGroups}
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Box>
      <Box css={WORKFLOW_LIBRARY_WIDE_ONLY_CSS} flexShrink={0}>
        <Menu.Root closeOnSelect={false} ids={viewIds} positioning={MENU_POSITIONING}>
          {/* Inside a dialog the tooltip must sit inside the menu trigger: wrapped the other way round, the menu never
              takes focus and the first pointer move onto it closes it. */}
          <Menu.Trigger asChild>
            <Tooltip content={t('workflowLibrary.viewOptions')} ids={viewIds}>
              <IconButton aria-label={t('workflowLibrary.viewOptions')} data-workflow-library-view variant="outline">
                {layout === 'list' ? <ListIcon /> : <LayoutGridIcon />}
              </IconButton>
            </Tooltip>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="12rem">{viewGroups}</MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Box>
      {/* The narrow layout's one menu for every order and view choice; it stays open across choices. */}
      <Box css={NARROW_ONLY_CSS} flexShrink={0}>
        <Menu.Root closeOnSelect={false} ids={narrowOptionsIds} positioning={MENU_POSITIONING}>
          <Menu.Trigger asChild>
            <Tooltip content={t('workflowLibrary.sortAndViewOptions')} ids={narrowOptionsIds}>
              <IconButton
                aria-label={t('workflowLibrary.sortAndViewOptions')}
                data-workflow-library-narrow-options
                variant="outline"
              >
                <SlidersHorizontalIcon />
              </IconButton>
            </Tooltip>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent maxH="var(--available-height)" minW="13rem" overflowY="auto">
                {orderGroups}
                <Menu.Separator />
                {viewGroups}
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Box>
      {onNewWorkflow ? (
        <Button flexShrink={0} onClick={onNewWorkflow}>
          <PlusIcon />
          {t('workflowLibrary.newWorkflow')}
        </Button>
      ) : null}
    </HStack>
  );
};
