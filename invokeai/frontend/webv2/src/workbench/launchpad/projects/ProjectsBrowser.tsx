import type { SystemStyleObject } from '@chakra-ui/react';

import { Alert, Box, Flex, Icon, SimpleGrid, Skeleton, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { Scrollable } from '@platform/ui/Scrollable';
import { Link } from '@tanstack/react-router';
import { refreshProjectLibrary, useProjectLibrarySelector } from '@workbench/projects/library';
import { useOpenProjectsSelector } from '@workbench/projects/openProjects';
import { FolderIcon, PlusIcon, SearchXIcon } from 'lucide-react';
import { useCallback, useEffectEvent, useLayoutEffect, useMemo, useRef } from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { ProjectGroupId, ProjectRowModel, ProjectSortId, ProjectsViewId } from './projectLibraryView';

import { ProjectCard } from './ProjectCard';
import { PROJECT_COVER_ASPECT_RATIO } from './ProjectCover';
import { buildProjectGroups, flattenProjectGroupsToRows } from './projectLibraryView';
import { ProjectRow } from './ProjectRow';
import {
  getProjectsGridRowHeight,
  PROJECT_GROUP_HEADER_HEIGHT_PX,
  PROJECT_ROW_HEIGHT_PX,
  PROJECTS_GRID_GAP_PX,
  useProjectsMetrics,
} from './projectsMetrics';

/**
 * The library itself: grouped, virtualized, and in whichever layout was chosen
 * last.
 *
 * Rows are the virtualization unit in both layouts — a grid row holds
 * `columnCount` cards, a list row holds one — so headings and items share a
 * single flat index space and grouping costs nothing extra. Row heights are
 * derived rather than measured, which is what lets the virtualizer place
 * everything without a layout pass.
 */

const MEASURE_SX: SystemStyleObject = {
  maxW: '6xl',
  mx: 'auto',
  px: { base: 4, md: 8 },
  w: 'full',
};

const GROUP_LABEL_KEY: Record<ProjectGroupId, string> = {
  all: 'projects.groups.all',
  month: 'projects.groups.month',
  older: 'projects.groups.older',
  open: 'projects.groups.open',
  pinned: 'projects.groups.pinned',
  today: 'projects.groups.today',
  week: 'projects.groups.week',
};

const SKELETON_GRID_COLUMNS = { base: 1, lg: 3, sm: 2 } as const;
const NO_MATCHES_ICON = <Icon as={SearchXIcon} />;
const NO_PROJECTS_ICON = <Icon as={FolderIcon} />;
const NEW_PROJECT_SEARCH = { new: true } as const;

export interface ProjectsBrowserProps {
  /**
   * When "today" starts, captured once by the page rather than read here —
   * reading the clock during render is impure and would rebucket unstably
   * across re-renders.
   */
  now: number;
  pinnedIds: readonly string[];
  searchTerm: string;
  sort: ProjectSortId;
  view: ProjectsViewId;
  onClearSearch: () => void;
  onTogglePin: (projectId: string) => void;
}

export const ProjectsBrowser = ({
  now,
  onClearSearch,
  onTogglePin,
  pinnedIds,
  searchTerm,
  sort,
  view,
}: ProjectsBrowserProps) => {
  const { t } = useTranslation();
  const error = useProjectLibrarySelector((snapshot) => snapshot.error);
  const status = useProjectLibrarySelector((snapshot) => snapshot.status);
  const summaries = useProjectLibrarySelector((snapshot) => snapshot.summaries);
  const openProjectIds = useOpenProjectsSelector((snapshot) => snapshot.ids);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const { columnCount, measureRef, widthPx } = useProjectsMetrics(view);

  const rows = useMemo(
    () =>
      flattenProjectGroupsToRows(
        buildProjectGroups({
          now,
          openProjectIds: openProjectIds ?? [],
          pinnedIds,
          searchTerm,
          sort,
          summaries,
        }),
        columnCount
      ),
    [columnCount, now, openProjectIds, pinnedIds, searchTerm, sort, summaries]
  );

  const itemRowHeight =
    view === 'grid'
      ? getProjectsGridRowHeight({ aspectRatio: PROJECT_COVER_ASPECT_RATIO, columnCount, widthPx })
      : PROJECT_ROW_HEIGHT_PX;

  const estimateSize = useCallback(
    (index: number) => (rows[index]?.kind === 'header' ? PROJECT_GROUP_HEADER_HEIGHT_PX : itemRowHeight),
    [itemRowHeight, rows]
  );
  const getItemKey = useCallback(
    (index: number) => {
      const row = rows[index];

      if (!row) {
        return index;
      }

      return row.kind === 'header'
        ? `header:${row.group}`
        : `${row.group}:${row.projects.map((project) => project.id).join(',')}`;
    },
    [rows]
  );
  const getScrollElement = useCallback(() => scrollRef.current, []);

  const virtualizer = useVirtualizer({
    count: rows.length,
    estimateSize,
    getItemKey,
    getScrollElement,
    overscan: 4,
  });

  // The virtualizer caches measurements and will not re-read `estimateSize` on
  // its own, so switching layouts or resizing the panel has to invalidate it —
  // otherwise list rows keep the taller grid row's spacing.
  const measureVirtualizer = useEffectEvent(() => {
    virtualizer.measure();
  });

  useLayoutEffect(() => {
    measureVirtualizer();
  }, [itemRowHeight, view]);

  const handleRetry = useCallback(() => void refreshProjectLibrary(), []);
  const isFirstLoad = summaries.length === 0 && (status === 'idle' || status === 'loading');
  const isSearching = searchTerm.trim().length > 0;

  return (
    <Scrollable flex="1" h="full" label={t('launchpad.sections.projects')} minH="0" viewportRef={scrollRef}>
      <Box css={MEASURE_SX} pb="8" ref={measureRef}>
        {status === 'error' ? (
          <Alert.Root borderRadius="md" mb="4" size="sm" status="error">
            <Alert.Indicator />
            <Alert.Title flex="1" fontSize="xs">
              {error ?? t('projects.failedToLoad')}
            </Alert.Title>
            <Button size="2xs" variant="outline" onClick={handleRetry}>
              {t('common.retry')}
            </Button>
          </Alert.Root>
        ) : null}

        {isFirstLoad ? (
          <SimpleGrid columns={SKELETON_GRID_COLUMNS} gap="4">
            <Skeleton minH="52" rounded="lg" />
            <Skeleton minH="52" rounded="lg" />
            <Skeleton minH="52" rounded="lg" />
          </SimpleGrid>
        ) : rows.length === 0 ? (
          <ProjectsEmptyState isSearching={isSearching} searchTerm={searchTerm} onClearSearch={onClearSearch} />
        ) : (
          // No "new project" cell here: the page header already carries that
          // action, and a lone dashed tile in a four-column row reads as a
          // broken grid. The empty state, where it is the only thing to do,
          // still offers it.
          <Box height={`${virtualizer.totalSize}px`} position="relative" w="full">
            {virtualizer.virtualItems.map((virtualRow) => {
              const row = rows[virtualRow.index];

              if (!row) {
                return null;
              }

              return (
                <VirtualProjectsRow
                  columnCount={columnCount}
                  key={virtualRow.key}
                  pinnedIds={pinnedIds}
                  row={row}
                  view={view}
                  virtualStart={virtualRow.start}
                  onTogglePin={onTogglePin}
                />
              );
            })}
          </Box>
        )}
      </Box>
    </Scrollable>
  );
};

const ABSOLUTE_ROW_SX: SystemStyleObject = { insetInline: 0, position: 'absolute', top: 0 };

const VirtualProjectsRow = ({
  columnCount,
  onTogglePin,
  pinnedIds,
  row,
  view,
  virtualStart,
}: {
  columnCount: number;
  pinnedIds: readonly string[];
  row: ProjectRowModel;
  view: ProjectsViewId;
  virtualStart: number;
  onTogglePin: (projectId: string) => void;
}) => {
  const { t } = useTranslation();
  const transform = `translateY(${virtualStart}px)`;

  if (row.kind === 'header') {
    return (
      <Flex align="baseline" css={ABSOLUTE_ROW_SX} gap="2" pb="2" pt="3" transform={transform}>
        <Text fontSize="xs" fontWeight="700">
          {t(GROUP_LABEL_KEY[row.group])}
        </Text>
        {/* `fg.subtle` only reaches 4.11:1 at this size — `fg.muted` clears 4.5:1. */}
        <Text color="fg.muted" fontSize="2xs">
          {row.count}
        </Text>
      </Flex>
    );
  }

  return (
    <Box css={ABSOLUTE_ROW_SX} transform={transform}>
      {view === 'grid' ? (
        <SimpleGrid columns={columnCount} gap={`${PROJECTS_GRID_GAP_PX}px`}>
          {row.projects.map((project) => (
            <ProjectCard
              isPinned={pinnedIds.includes(project.id)}
              key={project.id}
              summary={project}
              onTogglePin={onTogglePin}
            />
          ))}
        </SimpleGrid>
      ) : (
        row.projects.map((project) => (
          // The wrapper owns the virtualizer's full row pitch. No divider: the
          // row's own rounded hover fill separates the items, and a hairline
          // running under a rounded fill only cuts across its corners.
          <Box h={`${PROJECT_ROW_HEIGHT_PX}px`} key={project.id}>
            <ProjectRow isPinned={pinnedIds.includes(project.id)} summary={project} onTogglePin={onTogglePin} />
          </Box>
        ))
      )}
    </Box>
  );
};

/**
 * Both empty states align to `start`, on the same axis as the page heading,
 * the description and the search field. A centered block under a left-aligned
 * page shares an edge with nothing above it.
 *
 * The action is a button, not the dashed tile this used to render. That tile
 * was a grid cell with no grid around it — floating at an arbitrary width, and
 * shown even in list view — and it repeated a call to action the header
 * already carries. A button matches what the no-matches state beside it does.
 */
const ProjectsEmptyState = ({
  isSearching,
  onClearSearch,
  searchTerm,
}: {
  isSearching: boolean;
  searchTerm: string;
  onClearSearch: () => void;
}) => {
  const { t } = useTranslation();

  if (isSearching) {
    return (
      <EmptyState
        align="start"
        description={t('projects.noSearchMatchesDescription')}
        icon={NO_MATCHES_ICON}
        title={t('projects.noSearchMatches', { search: searchTerm.trim() })}
      >
        <Button size="xs" variant="outline" onClick={onClearSearch}>
          {t('common.clearSearch')}
        </Button>
      </EmptyState>
    );
  }

  return (
    <EmptyState
      align="start"
      description={t('projects.noSavedProjectsDescription')}
      icon={NO_PROJECTS_ICON}
      title={t('projects.noSavedProjects')}
    >
      {/* Outline, not solid: the header already carries this action as the
          page's one accent, and repeating that weight a few hundred pixels
          away doubles the loudest thing on an otherwise quiet screen without
          adding a choice. */}
      <Button asChild size="xs" variant="outline">
        <Link search={NEW_PROJECT_SEARCH} to="/app">
          <Icon as={PlusIcon} boxSize="3.5" />
          {t('projects.newProject')}
        </Link>
      </Button>
    </EmptyState>
  );
};
