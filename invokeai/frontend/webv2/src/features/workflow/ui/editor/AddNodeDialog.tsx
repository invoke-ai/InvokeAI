import type { InvocationTemplate } from '@features/workflow/contracts';
import type { AddNodeConnectionFilter } from '@features/workflow/ui/workflowUiStore';

import { Badge, Box, Dialog, HStack, Icon, Input, Portal, ScrollArea, Stack, Text } from '@chakra-ui/react';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import { useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import { getCompatibleInputTemplate, getCompatibleOutputTemplate } from '@features/workflow/utility';
import { IconButton, Tooltip } from '@platform/ui';
import { ChevronDownIcon, ChevronsDownUpIcon, ChevronsUpDownIcon, HammerIcon } from 'lucide-react';
import {
  startTransition,
  useCallback,
  useEffect,
  useEffectEvent,
  useMemo,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';

/**
 * Command-palette-style node picker: a centered search dialog whose results
 * are grouped into collapsible categories (with counts), mirroring the legacy
 * editor's add-node menu. Beta nodes carry the hammer icon. The two UI-only
 * nodes (Notes, Current Image) lead as a "Utility" group. No result cap —
 * searching auto-expands every group; idle shows all categories collapsed.
 */

const UTILITY_CATEGORY = 'Utility';
const CATEGORY_ROW_HEIGHT_PX = 28;
const NODE_ROW_HEIGHT_PX = 44;
const RESULT_LIST_ID = 'add-node-dialog-results';
const ROW_HOVER_PROPS = { bg: 'bg.emphasized' };
const VIRTUALIZER_INITIAL_RECT = { height: 384, width: 0 };

const toCategoryLabel = (value: string): string =>
  value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .trim();

const matchesSearch = (template: InvocationTemplate, terms: string[]): boolean => {
  const haystack = `${template.title} ${template.type} ${template.tags.join(' ')} ${template.category}`.toLowerCase();

  return terms.every((term) => haystack.includes(term));
};

const isCompatibleConnectionTemplate = (
  template: InvocationTemplate,
  connectionFilter: AddNodeConnectionFilter | null
): boolean => {
  if (!connectionFilter) {
    return true;
  }

  if (connectionFilter.kind === 'source') {
    return getCompatibleInputTemplate(template, connectionFilter.sourceType) !== null;
  }

  return getCompatibleOutputTemplate(template, connectionFilter.targetType) !== null;
};

const getConnectionFilterName = (connectionFilter: AddNodeConnectionFilter): string => {
  if (connectionFilter.kind === 'source') {
    return connectionFilter.sourceType?.name ?? 'connector';
  }

  return connectionFilter.targetType?.name ?? 'connector';
};

interface NodeRow {
  description: string;
  isBeta: boolean;
  key: string;
  nodePack: string;
  onAdd: () => void;
  title: string;
}

interface CategoryGroup {
  label: string;
  rows: NodeRow[];
}

type ResultRow =
  | { group: CategoryGroup; id: string; isExpanded: boolean; kind: 'category' }
  | { groupLabel: string; id: string; kind: 'node'; row: NodeRow };

const getResultRowId = (index: number): string => `${RESULT_LIST_ID}-${index}`;

const NodeResultRow = ({
  id,
  isActive,
  onActive,
  row,
}: {
  id: string;
  isActive: boolean;
  onActive: () => void;
  row: NodeRow;
}) => (
  <Box
    id={id}
    as="button"
    aria-level={2}
    aria-selected={isActive}
    bg={isActive ? 'bg.emphasized' : undefined}
    role="treeitem"
    tabIndex={-1}
    _hover={ROW_HOVER_PROPS}
    ps="5"
    pe="1.5"
    py="1.5"
    rounded="md"
    textAlign="start"
    w="full"
    onClick={row.onAdd}
    onMouseEnter={onActive}
    cursor="pointer"
  >
    <HStack gap="2" justify="space-between" alignItems="start">
      <Stack gap="0" minW="0">
        <HStack gap="1.5" minW="0">
          {row.isBeta ? (
            <Tooltip content="Beta node — may change in the future">
              <Icon as={HammerIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
            </Tooltip>
          ) : null}
          <Text fontSize="xs" fontWeight="600" truncate>
            {row.title}
          </Text>
        </HStack>
        {row.description ? (
          <Text color="fg.subtle" fontSize="2xs" lineClamp={2} lineHeight="1.4">
            {row.description}
          </Text>
        ) : null}
      </Stack>
      <Badge size="xs" variant="outline" fontFamily="mono">
        {row.nodePack}
      </Badge>
    </HStack>
  </Box>
);

const CategoryHeaderRow = ({
  group,
  id,
  isActive,
  isExpanded,
  onActive,
  onToggle,
}: {
  group: CategoryGroup;
  id: string;
  isActive: boolean;
  isExpanded: boolean;
  onActive: () => void;
  onToggle: (label: string) => void;
}) => {
  const toggle = useCallback(() => onToggle(group.label), [group.label, onToggle]);

  return (
    <Box
      id={id}
      as="button"
      aria-expanded={isExpanded}
      aria-level={1}
      aria-selected={isActive}
      bg={isActive ? 'bg.emphasized' : undefined}
      role="treeitem"
      tabIndex={-1}
      _hover={ROW_HOVER_PROPS}
      cursor="pointer"
      ps="1"
      pe="2"
      py="1"
      textAlign="start"
      w="full"
      onClick={toggle}
      onMouseEnter={onActive}
      rounded="md"
    >
      <HStack gap="1.5">
        <Icon
          as={ChevronDownIcon}
          boxSize="3"
          color="fg.subtle"
          flexShrink={0}
          transform={isExpanded ? 'rotate(0deg)' : 'rotate(-90deg)'}
          transition="transform var(--wb-motion-duration-fast) ease-out"
        />
        <Text flex="1" fontSize="xs" fontWeight="700">
          {group.label}
        </Text>
        <Badge size="sm" variant="surface" fontFamily="mono">
          {group.rows.length}
        </Badge>
      </HStack>
    </Box>
  );
};

export const AddNodeDialog = ({
  connectionFilter,
  isOpen,
  onAddCurrentImage,
  onAddConnector,
  onAddNode,
  onAddNote,
  onOpenChange,
}: {
  connectionFilter: AddNodeConnectionFilter | null;
  isOpen: boolean;
  onAddCurrentImage: () => void;
  onAddConnector: () => void;
  onAddNode: (template: InvocationTemplate) => void;
  onAddNote: () => void;
  onOpenChange: (isOpen: boolean) => void;
}) => {
  const { registerModalHotkeyLayer } = useWorkflowUi();
  const onDialogOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onOpenChange(false);
      }
    },
    [onOpenChange]
  );

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    return registerModalHotkeyLayer('workflow-add-node');
  }, [isOpen, registerModalHotkeyLayer]);

  return (
    <Dialog.Root
      lazyMount
      open={isOpen}
      placement="center"
      scrollBehavior="inside"
      size="md"
      unmountOnExit
      onOpenChange={onDialogOpenChange}
    >
      <AddNodeDialogContent
        connectionFilter={connectionFilter}
        onAddCurrentImage={onAddCurrentImage}
        onAddConnector={onAddConnector}
        onAddNode={onAddNode}
        onAddNote={onAddNote}
        onOpenChange={onOpenChange}
      />
    </Dialog.Root>
  );
};

const AddNodeDialogContent = ({
  connectionFilter,
  onAddCurrentImage,
  onAddConnector,
  onAddNode,
  onAddNote,
  onOpenChange,
}: {
  connectionFilter: AddNodeConnectionFilter | null;
  onAddCurrentImage: () => void;
  onAddConnector: () => void;
  onAddNode: (template: InvocationTemplate) => void;
  onAddNote: () => void;
  onOpenChange: (isOpen: boolean) => void;
}) => {
  const error = useInvocationTemplatesSelector((snapshot) => snapshot.error);
  const status = useInvocationTemplatesSelector((snapshot) => snapshot.status);
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(() => new Set());
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(null);
  // While searching, every matching group is force-expanded regardless of the
  // manual expand/collapse state, so results are never hidden behind a header.
  const isSearching = searchTerm.trim().length > 0;

  const close = useCallback(
    (added: boolean) => {
      onOpenChange(false);

      if (added) {
        setSearchTerm('');
      }
    },
    [onOpenChange]
  );

  const groups = useMemo<CategoryGroup[]>(() => {
    const terms = searchTerm.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const utilityRows: NodeRow[] = [
      {
        description: 'Route a connection through a compact pass-through handle.',
        isBeta: false,
        key: 'utility:connector',
        nodePack: 'invokeai',
        onAdd: () => {
          onAddConnector();
          close(true);
        },
        title: 'Connector',
      },
      ...(connectionFilter
        ? []
        : [
            {
              description: 'Annotate the workflow with a free-text note.',
              isBeta: false,
              key: 'utility:notes',
              nodePack: 'invokeai',
              onAdd: () => {
                onAddNote();
                close(true);
              },
              title: 'Notes',
            },
            {
              description: 'Show the latest generated image (and live progress) inside the graph.',
              isBeta: false,
              key: 'utility:current_image',
              nodePack: 'invokeai',
              onAdd: () => {
                onAddCurrentImage();
                close(true);
              },
              title: 'Current Image',
            },
          ]),
    ].filter((row) => terms.every((term) => row.title.toLowerCase().includes(term)));

    const byCategory = new Map<string, NodeRow[]>();

    for (const template of Object.values(templates)) {
      if (
        template.classification === 'internal' ||
        (terms.length > 0 && !matchesSearch(template, terms)) ||
        !isCompatibleConnectionTemplate(template, connectionFilter)
      ) {
        continue;
      }

      const label = toCategoryLabel(template.category || 'other');
      const row: NodeRow = {
        description: template.description,
        isBeta: template.classification === 'beta',
        key: `template:${template.type}`,
        nodePack: template.nodePack,
        onAdd: () => {
          onAddNode(template);
          close(true);
        },
        title: template.title,
      };

      byCategory.set(label, [...(byCategory.get(label) ?? []), row]);
    }

    const categoryGroups = [...byCategory.entries()]
      .map(([label, rows]) => ({
        label,
        rows: rows.sort((a, b) => a.title.localeCompare(b.title, undefined, { sensitivity: 'base' })),
      }))
      .sort((a, b) => a.label.localeCompare(b.label));

    return utilityRows.length > 0
      ? [{ label: UTILITY_CATEGORY, rows: utilityRows }, ...categoryGroups]
      : categoryGroups;
  }, [close, connectionFilter, onAddConnector, onAddCurrentImage, onAddNode, onAddNote, searchTerm, templates]);

  const totalCount = groups.reduce((sum, group) => sum + group.rows.length, 0);
  const isAllExpanded = groups.length > 0 && groups.every((group) => expandedCategories.has(group.label));

  const resultRows = useMemo<ResultRow[]>(() => {
    const rows: ResultRow[] = [];

    for (const group of groups) {
      const isExpanded = isSearching || expandedCategories.has(group.label);
      rows.push({ group, id: `category:${group.label}`, isExpanded, kind: 'category' });

      if (isExpanded) {
        for (const row of group.rows) {
          rows.push({ groupLabel: group.label, id: row.key, kind: 'node', row });
        }
      }
    }

    return rows;
  }, [expandedCategories, groups, isSearching]);
  const effectiveActiveIndex =
    resultRows.length === 0 ? null : activeIndex === null ? 0 : Math.min(activeIndex, resultRows.length - 1);
  const estimateVirtualRowSize = useCallback(
    (index: number) => (resultRows[index]?.kind === 'category' ? CATEGORY_ROW_HEIGHT_PX : NODE_ROW_HEIGHT_PX),
    [resultRows]
  );
  const getVirtualItemKey = useCallback((index: number) => resultRows[index]?.id ?? index, [resultRows]);
  const getVirtualScrollElement = useCallback(() => scrollElement, [scrollElement]);

  const virtualizer = useVirtualizer({
    count: resultRows.length,
    estimateSize: estimateVirtualRowSize,
    getItemKey: getVirtualItemKey,
    getScrollElement: getVirtualScrollElement,
    initialRect: VIRTUALIZER_INITIAL_RECT,
    overscan: 8,
  });
  const measureVirtualizer = useEffectEvent(() => {
    virtualizer.measure();
  });

  const toggleCategory = useCallback((label: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);

      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }

      return next;
    });
  }, []);

  const toggleAllCategories = useCallback(() => {
    startTransition(() => {
      setExpandedCategories((prev) => {
        const shouldCollapse = groups.length > 0 && groups.every((group) => prev.has(group.label));

        return shouldCollapse ? new Set() : new Set(groups.map((group) => group.label));
      });
    });
  }, [groups]);

  useEffect(() => {
    measureVirtualizer();
  }, [resultRows.length]);

  useEffect(() => {
    if (effectiveActiveIndex !== null) {
      virtualizer.scrollToIndex(effectiveActiveIndex, { align: 'auto' });
    }
  }, [effectiveActiveIndex, virtualizer]);

  const moveActiveIndex = useCallback(
    (direction: 1 | -1) => {
      setActiveIndex((prev) => {
        if (resultRows.length === 0) {
          return null;
        }

        if (prev === null) {
          return direction > 0 ? 0 : resultRows.length - 1;
        }

        return (prev + direction + resultRows.length) % resultRows.length;
      });
    },
    [resultRows.length]
  );

  const activateResultRow = useCallback(
    (row: ResultRow | undefined) => {
      if (!row) {
        return;
      }

      if (row.kind === 'category') {
        toggleCategory(row.group.label);
        return;
      }

      row.row.onAdd();
    },
    [toggleCategory]
  );

  const onSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        moveActiveIndex(1);
        return;
      }

      if (event.key === 'ArrowUp') {
        event.preventDefault();
        moveActiveIndex(-1);
        return;
      }

      if (event.key === 'Enter') {
        event.preventDefault();
        activateResultRow(effectiveActiveIndex === null ? undefined : resultRows[effectiveActiveIndex]);
      }
    },
    [activateResultRow, effectiveActiveIndex, moveActiveIndex, resultRows]
  );
  const onSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setSearchTerm(event.currentTarget.value),
    []
  );

  let body: ReactNode;

  if (status !== 'loaded') {
    body = (
      <Text color={status === 'error' ? 'fg.error' : 'fg.subtle'} fontSize="xs" px="1" py="4">
        {status === 'error' ? (error ?? 'Failed to load node definitions.') : 'Loading node definitions…'}
      </Text>
    );
  } else if (totalCount === 0) {
    body = (
      <Text color="fg.subtle" fontSize="xs" px="1" py="4" textAlign="center">
        {connectionFilter
          ? `No compatible ${getConnectionFilterName(connectionFilter)} nodes.`
          : 'No nodes match your search.'}
      </Text>
    );
  } else {
    body = (
      <Box h={`${virtualizer.totalSize}px`} position="relative" w="full">
        {virtualizer.virtualItems.map((virtualRow) => (
          <VirtualResultRow
            key={virtualRow.key}
            activeIndex={effectiveActiveIndex}
            measureElement={virtualizer.measureElement}
            resultRows={resultRows}
            virtualIndex={virtualRow.index}
            virtualStart={virtualRow.start}
            onActiveIndexChange={setActiveIndex}
            onToggleCategory={toggleCategory}
          />
        ))}
      </Box>
    );
  }

  return (
    <Portal>
      <Dialog.Backdrop />
      <Dialog.Positioner>
        <Dialog.Content h="min(512px, calc(100dvh - 4rem))">
          <Dialog.Body display="flex" minH="0" p="3">
            <Stack gap="2" flex="1" minH="0">
              <HStack gap="2" flexShrink={0}>
                <Input
                  autoFocus
                  aria-activedescendant={
                    effectiveActiveIndex === null ? undefined : getResultRowId(effectiveActiveIndex)
                  }
                  aria-controls={RESULT_LIST_ID}
                  aria-expanded="true"
                  aria-label="Search for nodes"
                  flex="1"
                  placeholder={
                    connectionFilter
                      ? `Search compatible ${getConnectionFilterName(connectionFilter)} nodes…`
                      : 'Search for nodes…'
                  }
                  role="combobox"
                  size="sm"
                  value={searchTerm}
                  onChange={onSearchChange}
                  onKeyDown={onSearchKeyDown}
                />
                <Tooltip content={isAllExpanded ? 'Collapse All' : 'Expand All'}>
                  <IconButton
                    aria-label={isAllExpanded ? 'Collapse all categories' : 'Expand all categories'}
                    size="sm"
                    variant="ghost"
                    onClick={toggleAllCategories}
                  >
                    <Icon as={isAllExpanded ? ChevronsUpDownIcon : ChevronsDownUpIcon} />
                  </IconButton>
                </Tooltip>
              </HStack>
              <ScrollArea.Root flex="1" minH="0" size="xs" variant="hover" w="full">
                <ScrollArea.Viewport ref={setScrollElement} aria-label="Node search results" h="full" w="full">
                  <ScrollArea.Content id={RESULT_LIST_ID} role="tree" w="full">
                    {body}
                  </ScrollArea.Content>
                </ScrollArea.Viewport>
                <ScrollArea.Scrollbar>
                  <ScrollArea.Thumb />
                </ScrollArea.Scrollbar>
              </ScrollArea.Root>
            </Stack>
          </Dialog.Body>
        </Dialog.Content>
      </Dialog.Positioner>
    </Portal>
  );
};

const VirtualResultRow = ({
  activeIndex,
  measureElement,
  resultRows,
  virtualIndex,
  virtualStart,
  onActiveIndexChange,
  onToggleCategory,
}: {
  activeIndex: number | null;
  measureElement: (node: Element | null) => void;
  resultRows: ResultRow[];
  virtualIndex: number;
  virtualStart: number;
  onActiveIndexChange: (index: number) => void;
  onToggleCategory: (label: string) => void;
}) => {
  const row = resultRows[virtualIndex];
  const onActive = useCallback(() => onActiveIndexChange(virtualIndex), [onActiveIndexChange, virtualIndex]);

  if (!row) {
    return null;
  }

  const isActive = virtualIndex === activeIndex;
  const id = getResultRowId(virtualIndex);

  return (
    <Box
      ref={measureElement}
      data-index={virtualIndex}
      left="0"
      position="absolute"
      top="0"
      transform={`translateY(${virtualStart}px)`}
      w="full"
    >
      {row.kind === 'category' ? (
        <CategoryHeaderRow
          id={id}
          group={row.group}
          isActive={isActive}
          isExpanded={row.isExpanded}
          onActive={onActive}
          onToggle={onToggleCategory}
        />
      ) : (
        <NodeResultRow id={id} isActive={isActive} onActive={onActive} row={row.row} />
      )}
    </Box>
  );
};
