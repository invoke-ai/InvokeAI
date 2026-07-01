/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Box, Checkbox, Flex, HStack, Icon, Image, ScrollArea, Spinner, Stack, Text } from '@chakra-ui/react';
import { Button, Row } from '@workbench/components/ui';
import { EmptyState } from '@workbench/components/ui/EmptyState';
import { MissingFileBadge, ModelBaseBadge, ModelFormatBadge } from '@workbench/launchpad/models/detail/ModelBadges';
import { getModelImageUrl } from '@workbench/models/api';
import {
  filterModels,
  flattenGroupsToRows,
  groupModelsByType,
  type ModelLibraryFilters,
} from '@workbench/models/library';
import { useModelsSelector } from '@workbench/models/modelsStore';
import { formatBytes } from '@workbench/models/taxonomy';
import { getLibraryScrollOffset, openModelManagerTab, saveLibraryScrollOffset } from '@workbench/models/uiStore';
import { ArrowRightIcon, BoxIcon, CircleAlert } from 'lucide-react';
import { memo, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import { ModelRowContextMenu, type ModelContextMenuTarget } from './ModelRowContextMenu';

const HEADER_ROW_HEIGHT_PX = 30;
const MODEL_ROW_HEIGHT_PX = 56;

/**
 * Virtualized, grouped model library. Group headers and model rows share one
 * flat virtualized list (smooth at thousands of models). The current group's
 * header is rendered as a pinned overlay above the scroll viewport — CSS
 * sticky positioning is unreliable inside ScrollArea's content wrapper — and
 * the virtual content lives in ScrollArea.Content so the scrollbar thumb is
 * measured correctly. Rows carry a thumbnail, a right-click action menu, and —
 * when `onToggleSelected` is provided — bulk-select checkboxes.
 */
export const ModelLibraryList = ({
  activeModelKey,
  filters,
  instanceId,
  onActivate,
  onToggleSelected,
  selectedKeys,
}: {
  activeModelKey: string | null;
  filters: ModelLibraryFilters;
  /** Identifies this list's scroll-offset slot ('panel', 'manager', ...). */
  instanceId: string;
  onActivate: (modelKey: string) => void;
  /** Present in the manager library: enables bulk-select checkboxes. */
  onToggleSelected?: (modelKey: string) => void;
  selectedKeys?: ReadonlySet<string>;
}) => {
  const { t } = useTranslation();
  const { coverImageVersions, error, missingModelKeys, models, status } = useModelsSelector(
    (snapshot) => ({
      coverImageVersions: snapshot.coverImageVersions,
      error: snapshot.error,
      missingModelKeys: snapshot.missingModelKeys,
      models: snapshot.models,
      status: snapshot.status,
    }),
    (left, right) =>
      left.coverImageVersions === right.coverImageVersions &&
      left.error === right.error &&
      left.missingModelKeys === right.missingModelKeys &&
      left.models === right.models &&
      left.status === right.status
  );
  const deferredFilters = useDeferredValue(filters);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [contextMenuTarget, setContextMenuTarget] = useState<ModelContextMenuTarget | null>(null);
  const openAddModels = () => {
    openModelManagerTab('add');
  };
  const handleContextMenu = useCallback(
    (modelKey: string, x: number, y: number) => setContextMenuTarget({ modelKey, x, y }),
    []
  );

  const rows = useMemo(
    () => flattenGroupsToRows(groupModelsByType(filterModels(models, deferredFilters, missingModelKeys))),
    [deferredFilters, missingModelKeys, models]
  );

  const headerIndexes = useMemo(() => rows.flatMap((row, index) => (row.kind === 'header' ? [index] : [])), [rows]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    estimateSize: (index) => (rows[index]?.kind === 'header' ? HEADER_ROW_HEIGHT_PX : MODEL_ROW_HEIGHT_PX),
    getItemKey: (index) => {
      const row = rows[index];

      return row?.kind === 'model' ? row.model.key : `header:${row?.group.type ?? index}`;
    },
    getScrollElement: () => scrollRef.current,
    initialOffset: () => getLibraryScrollOffset(instanceId),
    overscan: 8,
  });
  const scrollOffsetRef = useRef(virtualizer.scrollOffset ?? 0);

  useEffect(() => {
    scrollOffsetRef.current = virtualizer.scrollOffset ?? scrollOffsetRef.current;
  }, [virtualizer.scrollOffset]);

  // Persist the offset so switching tabs/regions and coming back lands here.
  useEffect(
    () => () => {
      saveLibraryScrollOffset(instanceId, scrollOffsetRef.current);
    },
    [instanceId]
  );

  if (status === 'loading' || status === 'idle') {
    return (
      <Flex align="center" h="full" justify="center" py="8">
        <Spinner color="fg.subtle" size="sm" />
      </Flex>
    );
  }

  if (status === 'error') {
    return <EmptyState title={t('models.couldNotLoad')} description={error} icon={<Icon as={CircleAlert} />} danger />;
  }

  if (rows.length === 0) {
    return (
      <EmptyState
        title={models.length === 0 ? t('models.noneInstalled') : t('models.noneMatchFilters')}
        description={
          models.length === 0 ? t('models.noneInstalledDescription') : t('models.noneMatchFiltersDescription')
        }
        icon={<Icon as={CircleAlert} />}
      >
        <Button onClick={openAddModels} size="sm">
          {t('models.addModels')}
          <Icon as={ArrowRightIcon} />
        </Button>
      </EmptyState>
    );
  }

  const firstVisibleIndex = virtualizer.virtualItems[0]?.index ?? 0;
  const pinnedHeaderIndex = headerIndexes.reduce<number | null>(
    (pinnedIndex, index) => (firstVisibleIndex >= index ? index : pinnedIndex),
    headerIndexes[0] ?? null
  );
  const pinnedRow = pinnedHeaderIndex === null ? null : rows[pinnedHeaderIndex];

  return (
    <Box flex="1" minH="0" position="relative" w="full">
      <ScrollArea.Root h="full" size="xs" variant="hover" w="full">
        <ScrollArea.Viewport ref={scrollRef} aria-label={t('models.library')} h="full" w="full">
          <ScrollArea.Content w="full">
            <Box h={`${virtualizer.totalSize}px`} position="relative" w="full">
              {virtualizer.virtualItems.map((virtualRow) => {
                const row = rows[virtualRow.index];

                if (!row) {
                  return null;
                }

                return row.kind === 'header' ? (
                  <VirtualHeaderRow
                    key={virtualRow.key}
                    count={row.group.models.length}
                    label={row.group.label}
                    virtualStart={virtualRow.start}
                  />
                ) : (
                  <VirtualModelRow
                    key={virtualRow.key}
                    base={row.model.base}
                    coverImage={row.model.cover_image}
                    fileSize={row.model.file_size}
                    format={row.model.format}
                    imageVersion={coverImageVersions[row.model.key]}
                    isActive={row.model.key === activeModelKey}
                    isMissing={missingModelKeys.has(row.model.key)}
                    isSelected={selectedKeys?.has(row.model.key) ?? false}
                    modelKey={row.model.key}
                    name={row.model.name}
                    virtualStart={virtualRow.start}
                    onActivate={onActivate}
                    onContextMenu={handleContextMenu}
                    onToggleSelected={onToggleSelected}
                  />
                );
              })}
            </Box>
          </ScrollArea.Content>
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar>
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>
      {/* Pinned copy of the current group's header, above the viewport.
          aria-hidden: it duplicates a header already in the list. */}
      {pinnedRow?.kind === 'header' ? (
        <Box aria-hidden bg="bg" left="0" pointerEvents="none" position="absolute" right="0" top="0" zIndex={1} px="3">
          <GroupHeader count={pinnedRow.group.models.length} label={pinnedRow.group.label} />
        </Box>
      ) : null}
      <ModelRowContextMenu target={contextMenuTarget} onClose={() => setContextMenuTarget(null)} />
    </Box>
  );
};

const GroupHeader = ({ count, label }: { count: number; label: string }) => (
  <HStack bg="bg.inset" gap="1.5" h={`${HEADER_ROW_HEIGHT_PX}px`} pt="2" w="full">
    <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
      {label}
    </Text>
    <Text color="fg.subtle" fontSize="2xs">
      {count}
    </Text>
  </HStack>
);

interface VirtualHeaderRowProps {
  count: number;
  label: string;
  virtualStart: number;
}

const VirtualHeaderRow = memo(function VirtualHeaderRow({ count, label, virtualStart }: VirtualHeaderRowProps) {
  return (
    <Box left="0" position="absolute" top="0" transform={`translateY(${virtualStart}px)`} w="full" px="3">
      <GroupHeader count={count} label={label} />
    </Box>
  );
});

interface ModelRowProps {
  base: Parameters<typeof ModelBaseBadge>[0]['base'];
  coverImage?: string | null;
  fileSize: number;
  format: Parameters<typeof ModelFormatBadge>[0]['format'];
  imageVersion?: number;
  isActive: boolean;
  isMissing: boolean;
  isSelected: boolean;
  modelKey: string;
  name: string;
  onActivate: (modelKey: string) => void;
  onContextMenu: (modelKey: string, x: number, y: number) => void;
  onToggleSelected?: (modelKey: string) => void;
}

interface VirtualModelRowProps extends ModelRowProps {
  virtualStart: number;
}

const VirtualModelRow = memo(function VirtualModelRow({ virtualStart, ...rowProps }: VirtualModelRowProps) {
  return (
    <Box left="0" position="absolute" top="0" transform={`translateY(${virtualStart}px)`} w="full" px="3">
      <ModelRow {...rowProps} />
    </Box>
  );
});

const ModelRow = memo(function ModelRow({
  base,
  coverImage,
  fileSize,
  format,
  imageVersion,
  isActive,
  isMissing,
  isSelected,
  modelKey,
  name,
  onActivate,
  onContextMenu,
  onToggleSelected,
}: ModelRowProps) {
  return (
    <Row
      active={isActive ? 'accent' : isSelected ? 'muted' : 'none'}
      aria-current={isActive || undefined}
      h={`${MODEL_ROW_HEIGHT_PX - 4}px`}
      mb="1"
      minW="0"
      px="2"
      role="button"
      rounded="md"
      tabIndex={0}
      _focusVisible={{ boxShadow: 'inset 0 0 0 2px {colors.accent.solid}', outline: 'none' }}
      onClick={() => onActivate(modelKey)}
      onContextMenu={(event) => {
        event.preventDefault();
        onContextMenu(modelKey, event.clientX, event.clientY);
      }}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onActivate(modelKey);
        }
      }}
    >
      {onToggleSelected ? (
        <Checkbox.Root
          aria-label={`Select ${name}`}
          checked={isSelected}
          colorPalette="accent"
          size="xs"
          onCheckedChange={() => onToggleSelected(modelKey)}
          onClick={(event) => event.stopPropagation()}
        >
          <Checkbox.HiddenInput />
          <Checkbox.Control />
        </Checkbox.Root>
      ) : null}
      {/* Keyed by version so a stale load error clears when the image changes. */}
      <ModelRowThumbnail
        key={`${modelKey}:${imageVersion ?? 0}`}
        coverImage={coverImage}
        imageVersion={imageVersion}
        modelKey={modelKey}
      />
      <Stack flex="1" gap="0.5" minW="0">
        <Text fontSize="xs" fontWeight="600" truncate>
          {name}
        </Text>
        <HStack gap="1" minW="0" wrap="wrap">
          <ModelBaseBadge base={base} />
          <ModelFormatBadge format={format} />
          {isMissing ? <MissingFileBadge /> : null}
        </HStack>
      </Stack>
      <Text color={isActive ? 'accent.solid' : 'fg.subtle'} flexShrink={0} fontSize="2xs">
        {formatBytes(fileSize)}
      </Text>
    </Row>
  );
});

const ModelRowThumbnail = ({
  coverImage,
  imageVersion,
  modelKey,
}: {
  coverImage?: string | null;
  imageVersion?: number;
  modelKey: string;
}) => {
  // The cover_image marker can be stale; fall back to the icon on load error.
  const [hasImageError, setHasImageError] = useState(false);

  if (!coverImage || hasImageError) {
    return (
      <Flex
        align="center"
        bg="bg.emphasized"
        borderColor="border.subtle"
        borderWidth="1px"
        boxSize="9"
        color="fg.subtle"
        flexShrink={0}
        justify="center"
        rounded="md"
      >
        <Icon as={BoxIcon} boxSize="4" />
      </Flex>
    );
  }

  return (
    <Image
      alt=""
      boxSize="9"
      fit="cover"
      flexShrink={0}
      loading="lazy"
      rounded="md"
      src={getModelImageUrl(modelKey, imageVersion ? String(imageVersion) : undefined)}
      onError={() => setHasImageError(true)}
    />
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
