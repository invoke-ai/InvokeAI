import { Flex, Icon, Input, InputGroup, Menu, Portal, SegmentGroup } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { ArrowUpDownIcon, CheckIcon, LayoutGridIcon, ListIcon, SearchIcon, XIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/* eslint-disable react-perf/jsx-no-jsx-as-prop */
import type { ProjectSortId, ProjectsViewId } from './projectLibraryView';

import { isProjectsViewId, PROJECT_SORT_IDS, PROJECTS_VIEW_IDS } from './projectLibraryView';

/**
 * Search, ordering, and layout for the project library.
 *
 * The row spans the page's measure rather than packing to the left: search
 * holds the left edge under the heading, and the controls sit on the right
 * edge under the header's buttons. That gives the top of the page two vertical
 * lines instead of one line and a drifting cluster.
 *
 * The layout choice is a `SegmentGroup`, not two adjacent icon buttons. Two
 * buttons where one happens to be highlighted read as two actions, one of
 * which is somehow active; a segmented control reads as one control with two
 * positions, which is what it is. It also inherits the design system's
 * concentric radii — `radii.sm` items inside a `radii.md` track.
 */

const MENU_POSITIONING = { placement: 'bottom-end' } as const;
const SEARCH_ICON = <Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />;

const SORT_LABEL_KEY: Record<ProjectSortId, string> = {
  created: 'projects.sort.created',
  edited: 'projects.sort.edited',
  name: 'projects.sort.name',
};

const VIEW_LABEL_KEY: Record<ProjectsViewId, string> = {
  grid: 'projects.gridView',
  list: 'projects.listView',
};

const VIEW_ICON: Record<ProjectsViewId, typeof LayoutGridIcon> = {
  grid: LayoutGridIcon,
  list: ListIcon,
};

export const ProjectsToolbar = ({
  searchTerm,
  sort,
  view,
  onSearchTermChange,
  onSortChange,
  onViewChange,
}: {
  searchTerm: string;
  sort: ProjectSortId;
  view: ProjectsViewId;
  onSearchTermChange: (value: string) => void;
  onSortChange: (sort: ProjectSortId) => void;
  onViewChange: (view: ProjectsViewId) => void;
}) => {
  const { t } = useTranslation();

  const handleSearchInput = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => onSearchTermChange(event.target.value),
    [onSearchTermChange]
  );
  const handleClearSearch = useCallback(() => onSearchTermChange(''), [onSearchTermChange]);
  const handleSortSelect = useCallback(
    (details: { value: string }) => {
      if (PROJECT_SORT_IDS.includes(details.value as ProjectSortId)) {
        onSortChange(details.value as ProjectSortId);
      }
    },
    [onSortChange]
  );
  const handleViewChange = useCallback(
    (details: { value: string | null }) => {
      if (isProjectsViewId(details.value)) {
        onViewChange(details.value);
      }
    },
    [onViewChange]
  );

  return (
    <Flex align="center" gap="2" wrap="wrap">
      <InputGroup
        endElement={
          searchTerm ? (
            <IconButton aria-label={t('common.clearSearch')} size="2xs" variant="ghost" onClick={handleClearSearch}>
              <Icon as={XIcon} boxSize="3" />
            </IconButton>
          ) : undefined
        }
        flex="1"
        maxW="xs"
        minW="3xs"
        startElement={SEARCH_ICON}
      >
        <Input
          aria-label={t('projects.searchLabel')}
          placeholder={t('projects.searchPlaceholder')}
          size="xs"
          type="search"
          value={searchTerm}
          onChange={handleSearchInput}
        />
      </InputGroup>

      <Menu.Root positioning={MENU_POSITIONING} onSelect={handleSortSelect}>
        <Menu.Trigger asChild>
          <Button ms="auto" size="xs" variant="outline">
            <Icon as={ArrowUpDownIcon} boxSize="3.5" />
            {t(SORT_LABEL_KEY[sort])}
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="44">
              <Menu.ItemGroup>
                <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  {t('projects.sort.label')}
                </Menu.ItemGroupLabel>
                {PROJECT_SORT_IDS.map((id) => (
                  <Menu.Item key={id} value={id}>
                    <Icon as={CheckIcon} boxSize="3.5" opacity={id === sort ? 1 : 0} />
                    <Menu.ItemText>{t(SORT_LABEL_KEY[id])}</Menu.ItemText>
                  </Menu.Item>
                ))}
              </Menu.ItemGroup>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <SegmentGroup.Root aria-label={t('projects.viewLabel')} size="xs" value={view} onValueChange={handleViewChange}>
        <SegmentGroup.Indicator />
        {PROJECTS_VIEW_IDS.map((id) => (
          <SegmentGroup.Item key={id} px="2" value={id}>
            {/* The hidden radio is the control, so it carries the name — the
                visible label is an icon with no text of its own. */}
            <SegmentGroup.ItemHiddenInput aria-label={t(VIEW_LABEL_KEY[id])} />
            <SegmentGroup.ItemText display="flex">
              {/* The tooltip wraps the icon, never the item: `Tooltip.Trigger`
                  is `asChild` and merges its own `data-state` onto whatever it
                  clones, which on the item overwrites `data-state="checked"` —
                  the selected segment then styles as unselected and the
                  indicator measures 0×0. */}
              <Tooltip content={t(VIEW_LABEL_KEY[id])}>
                <Icon as={VIEW_ICON[id]} boxSize="3.5" />
              </Tooltip>
            </SegmentGroup.ItemText>
          </SegmentGroup.Item>
        ))}
      </SegmentGroup.Root>
    </Flex>
  );
};
