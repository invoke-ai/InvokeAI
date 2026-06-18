import type { ModelConfig } from '@workbench/models/types';

import { Box, Checkbox, Flex, HStack, Icon, Image, ScrollArea, Spinner, Stack, Text } from '@chakra-ui/react';
import { defaultRangeExtractor, useVirtualizer, type Range } from '@tanstack/react-virtual';
import { Button, Row } from '@workbench/components/ui';
import { EmptyState } from '@workbench/components/ui/EmptyState';
import { getModelImageUrl } from '@workbench/models/api';
import {
  filterModels,
  flattenGroupsToRows,
  groupModelsByType,
  type ModelLibraryFilters,
} from '@workbench/models/library';
import { useModelsSnapshot } from '@workbench/models/modelsStore';
import { formatBytes } from '@workbench/models/taxonomy';
import { getLibraryScrollOffset, openModelsCenterTab, saveLibraryScrollOffset } from '@workbench/models/uiStore';
import { ArrowRightIcon, BoxIcon, CircleAlert } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { ModelBadgeRow } from './ModelBadges';
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
  /** Identifies this list's scroll-offset slot ('panel', 'center', ...). */
  instanceId: string;
  onActivate: (model: ModelConfig) => void;
  /** Present in center view: enables bulk-select checkboxes. */
  onToggleSelected?: (model: ModelConfig) => void;
  selectedKeys?: ReadonlySet<string>;
}) => {
  const { coverImageVersions, error, missingModelKeys, models, status } = useModelsSnapshot();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [contextMenuTarget, setContextMenuTarget] = useState<ModelContextMenuTarget | null>(null);
  const openAddModels = () => {
    openModelsCenterTab('add');
  };

  const rows = useMemo(
    () => flattenGroupsToRows(groupModelsByType(filterModels(models, filters, missingModelKeys))),
    [filters, missingModelKeys, models]
  );

  const headerIndexes = useMemo(() => rows.flatMap((row, index) => (row.kind === 'header' ? [index] : [])), [rows]);
  const pinnedHeaderIndexRef = useRef<number | null>(null);

  // Track which group the viewport is inside. rangeExtractor sees the exact
  // visible range on every scroll, before render, so a ref read during render
  // is always current.
  const rangeExtractor = useCallback(
    (range: Range) => {
      pinnedHeaderIndexRef.current =
        [...headerIndexes].reverse().find((index) => range.startIndex >= index) ?? headerIndexes[0] ?? null;

      return defaultRangeExtractor(range);
    },
    [headerIndexes]
  );

  const virtualizer = useVirtualizer({
    count: rows.length,
    estimateSize: (index) => (rows[index]?.kind === 'header' ? HEADER_ROW_HEIGHT_PX : MODEL_ROW_HEIGHT_PX),
    getScrollElement: () => scrollRef.current,
    initialOffset: () => getLibraryScrollOffset(instanceId),
    overscan: 8,
    rangeExtractor,
  });
  const scrollOffsetRef = useRef(virtualizer.scrollOffset ?? 0);

  scrollOffsetRef.current = virtualizer.scrollOffset ?? scrollOffsetRef.current;

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
    return <EmptyState title="Could not load models" description={error} icon={<Icon as={CircleAlert} />} danger />;
  }

  if (rows.length === 0) {
    return (
      <EmptyState
        title={models.length === 0 ? 'No models installed' : 'No models match your filters'}
        description={
          models.length === 0 ? 'Use Add Models to install your first model.' : 'Try clearing the search or filters.'
        }
        icon={<Icon as={CircleAlert} />}
      >
        <Button onClick={openAddModels} size="sm">
          Add Models
          <Icon as={ArrowRightIcon} />
        </Button>
      </EmptyState>
    );
  }

  const pinnedHeaderIndex = pinnedHeaderIndexRef.current;
  const pinnedRow = pinnedHeaderIndex === null ? null : rows[pinnedHeaderIndex];

  return (
    <Box flex="1" minH="0" position="relative" w="full">
      <ScrollArea.Root h="full" size="xs" variant="hover" w="full">
        <ScrollArea.Viewport ref={scrollRef} aria-label="Model library" h="full" w="full">
          <ScrollArea.Content w="full">
            <Box h={`${virtualizer.getTotalSize()}px`} position="relative" w="full">
              {virtualizer.getVirtualItems().map((virtualRow) => {
                const row = rows[virtualRow.index];

                if (!row) {
                  return null;
                }

                return (
                  <Box
                    key={virtualRow.key}
                    left="0"
                    position="absolute"
                    top="0"
                    transform={`translateY(${virtualRow.start}px)`}
                    w="full"
                  >
                    {row.kind === 'header' ? (
                      <GroupHeader count={row.group.models.length} label={row.group.label} />
                    ) : (
                      <ModelRow
                        imageVersion={coverImageVersions[row.model.key]}
                        isActive={row.model.key === activeModelKey}
                        isMissing={missingModelKeys.has(row.model.key)}
                        isSelected={selectedKeys?.has(row.model.key) ?? false}
                        model={row.model}
                        onActivate={onActivate}
                        onContextMenu={(model, x, y) => setContextMenuTarget({ model, x, y })}
                        onToggleSelected={onToggleSelected}
                      />
                    )}
                  </Box>
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
        <Box aria-hidden bg="bg" left="0" pointerEvents="none" position="absolute" right="0" top="0" zIndex={1}>
          <GroupHeader count={pinnedRow.group.models.length} label={pinnedRow.group.label} />
        </Box>
      ) : null}
      <ModelRowContextMenu target={contextMenuTarget} onClose={() => setContextMenuTarget(null)} />
    </Box>
  );
};

const GroupHeader = ({ count, label }: { count: number; label: string }) => (
  <HStack bg="bg.inset" gap="1.5" h={`${HEADER_ROW_HEIGHT_PX}px`} pt="2" px="1" w="full">
    <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
      {label}
    </Text>
    <Text color="fg.subtle" fontSize="2xs">
      {count}
    </Text>
  </HStack>
);

const ModelRow = ({
  imageVersion,
  isActive,
  isMissing,
  isSelected,
  model,
  onActivate,
  onContextMenu,
  onToggleSelected,
}: {
  /** Cover-image cache-bust version from the models store. */
  imageVersion?: number;
  isActive: boolean;
  isMissing: boolean;
  isSelected: boolean;
  model: ModelConfig;
  onActivate: (model: ModelConfig) => void;
  onContextMenu: (model: ModelConfig, x: number, y: number) => void;
  onToggleSelected?: (model: ModelConfig) => void;
}) => (
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
    onClick={() => onActivate(model)}
    onContextMenu={(event) => {
      event.preventDefault();
      onContextMenu(model, event.clientX, event.clientY);
    }}
    onKeyDown={(event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onActivate(model);
      }
    }}
  >
    {onToggleSelected ? (
      <Checkbox.Root
        aria-label={`Select ${model.name}`}
        checked={isSelected}
        colorPalette="accent"
        size="xs"
        onCheckedChange={() => onToggleSelected(model)}
        onClick={(event) => event.stopPropagation()}
      >
        <Checkbox.HiddenInput />
        <Checkbox.Control />
      </Checkbox.Root>
    ) : null}
    {/* Keyed by version so a stale load error clears when the image changes. */}
    <ModelRowThumbnail key={imageVersion ?? 0} imageVersion={imageVersion} model={model} />
    <Stack flex="1" gap="0.5" minW="0">
      <Text fontSize="xs" fontWeight="600" truncate>
        {model.name}
      </Text>
      <ModelBadgeRow isMissing={isMissing} model={model} />
    </Stack>
    <Text color={isActive ? 'accent.solid' : 'fg.subtle'} flexShrink={0} fontSize="2xs">
      {formatBytes(model.file_size)}
    </Text>
  </Row>
);

const ModelRowThumbnail = ({ imageVersion, model }: { imageVersion?: number; model: ModelConfig }) => {
  // The cover_image marker can be stale; fall back to the icon on load error.
  const [hasImageError, setHasImageError] = useState(false);

  if (!model.cover_image || hasImageError) {
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
      src={getModelImageUrl(model.key, imageVersion ? String(imageVersion) : undefined)}
      onError={() => setHasImageError(true)}
    />
  );
};
