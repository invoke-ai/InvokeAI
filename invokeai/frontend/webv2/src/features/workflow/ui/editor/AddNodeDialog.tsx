import type { InvocationTemplate, InvocationTemplates } from '@features/workflow/contracts';
import type { WorkflowEdge, WorkflowNode } from '@features/workflow/core/types';
import type { AddNodeConnectionFilter } from '@features/workflow/ui/workflowUiStore';

import { Badge, Box, HStack, Icon, Input, Portal, ScrollArea, Spinner, Stack, Text } from '@chakra-ui/react';
import { ensureInvocationTemplatesLoaded, useInvocationTemplatesSelector } from '@features/workflow/react';
import { useWorkflowPreferencesSelector, useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import {
  getCompatibleInputTemplate,
  getCompatibleOutputTemplate,
  getFieldTypeLabel,
  LOOP_LINKAGE_FIELD,
  resolveConnectorSource,
} from '@features/workflow/utility';
import { isImeComposing } from '@platform/browser/imeComposition';
import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { Button, IconButton, Tooltip } from '@platform/ui';
import { Dialog } from '@platform/ui/Dialog';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ChevronDownIcon, ChevronsDownUpIcon, ChevronsUpDownIcon, HammerIcon } from 'lucide-react';
import {
  startTransition,
  useCallback,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

/**
 * Searching expands all node groups without capping results and ranks the closest names first; idle groups collapse
 * and UI-only nodes lead in Utility. With grouping off, results are one ranked list.
 */

const CATEGORY_ROW_HEIGHT_PX = 28;
const NODE_ROW_HEIGHT_PX = 44;
const RESULT_LIST_ID = 'add-node-dialog-results';
const ROW_HOVER_PROPS = { bg: 'bg.hover' };
const VIRTUALIZER_INITIAL_RECT = { height: 384, width: 0 };

const toCategoryLabel = (value: string): string =>
  value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .trim();

interface SearchQuery {
  terms: string[];
  /** The whole query, lowercased with single spaces. */
  text: string;
}

const parseSearchQuery = (searchTerm: string): SearchQuery => {
  const terms = searchTerm.trim().toLowerCase().split(/\s+/).filter(Boolean);

  return { terms, text: terms.join(' ') };
};

/**
 * Where a match lists, lower first: the exact name or type, a name prefix, every term in the name, then a match
 * anywhere else (type, tags, category). Null when a term matches nothing.
 */
const getSearchRank = (query: SearchQuery, name: string, otherText: string, type?: string): number | null => {
  if (query.terms.length === 0) {
    return 0;
  }

  const lowerName = name.toLowerCase();
  const haystack = `${lowerName} ${otherText.toLowerCase()}`;

  if (!query.terms.every((term) => haystack.includes(term))) {
    return null;
  }

  if (lowerName === query.text || type === query.text) {
    return 0;
  }

  if (lowerName.startsWith(query.text)) {
    return 1;
  }

  return query.terms.every((term) => lowerName.includes(term)) ? 2 : 3;
};

const getTemplateSearchRank = (template: InvocationTemplate, query: SearchQuery): number | null =>
  getSearchRank(
    query,
    template.title,
    `${template.type} ${template.tags.join(' ')} ${template.category}`,
    template.type
  );

const isCompatibleConnectionTemplate = (
  template: InvocationTemplate,
  connectionFilter: AddNodeConnectionFilter | null
): boolean => {
  if (!connectionFilter) {
    return true;
  }

  if (connectionFilter.kind === 'source') {
    if (connectionFilter.sourceHandle === LOOP_LINKAGE_FIELD) {
      return template.type === 'for_return' && template.inputs[LOOP_LINKAGE_FIELD] !== undefined;
    }

    return getCompatibleInputTemplate(template, connectionFilter.sourceType) !== null;
  }

  if (connectionFilter.targetHandle === LOOP_LINKAGE_FIELD) {
    return template.type === 'for' && template.outputs[LOOP_LINKAGE_FIELD] !== undefined;
  }

  return getCompatibleOutputTemplate(template, connectionFilter.targetType) !== null;
};

const isForIterationOutputConnection = (
  connectionFilter: AddNodeConnectionFilter | null,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  templates: InvocationTemplates
): boolean => {
  if (!connectionFilter || connectionFilter.kind !== 'source') {
    return false;
  }

  const sourceNode = nodes.find((node) => node.id === connectionFilter.sourceNodeId);
  const resolvedSource =
    sourceNode?.type === 'connector'
      ? resolveConnectorSource(sourceNode.id, nodes, edges, templates)
      : sourceNode?.type === 'invocation'
        ? { fieldName: connectionFilter.sourceHandle, nodeId: sourceNode.id }
        : null;

  if (!resolvedSource) {
    return false;
  }

  const resolvedSourceNode = nodes.find((node) => node.id === resolvedSource.nodeId);
  return (
    resolvedSourceNode?.type === 'invocation' &&
    resolvedSourceNode.data.type === 'for' &&
    templates[resolvedSourceNode.data.type]?.outputs[resolvedSource.fieldName]?.outputScope === 'iteration'
  );
};

/** The field type a pending connection needs; null when it starts at a connector whose type is still unresolved. */
const getConnectionFilterTypeLabel = (connectionFilter: AddNodeConnectionFilter): string | null => {
  const type = connectionFilter.kind === 'source' ? connectionFilter.sourceType : connectionFilter.targetType;

  return type ? getFieldTypeLabel(type) : null;
};

interface NodeRow {
  description: string;
  isBeta: boolean;
  isUtility: boolean;
  key: string;
  nodePack: string;
  onAdd: () => void;
  /** From `getSearchRank`; 0 while not searching. */
  rank: number;
  title: string;
}

const FOR_RETURN_KEY = 'template:for_return';

const compareRows = (a: NodeRow, b: NodeRow, shouldPromoteForReturn: boolean): number => {
  if (shouldPromoteForReturn && (a.key === FOR_RETURN_KEY) !== (b.key === FOR_RETURN_KEY)) {
    return a.key === FOR_RETURN_KEY ? -1 : 1;
  }

  if (a.rank !== b.rank) {
    return a.rank - b.rank;
  }

  if (a.isUtility !== b.isUtility) {
    return a.isUtility ? -1 : 1;
  }

  return a.title.localeCompare(b.title, undefined, { sensitivity: 'base' });
};

interface CategoryGroup {
  isUtility: boolean;
  label: string;
  rows: NodeRow[];
}

type ResultRow =
  | { group: CategoryGroup; id: string; isExpanded: boolean; kind: 'category' }
  | { id: string; kind: 'node'; level: 1 | 2; row: NodeRow };

const getResultRowId = (index: number): string => `${RESULT_LIST_ID}-${index}`;

const NodeResultRow = ({
  id,
  isActive,
  level,
  onActive,
  row,
}: {
  id: string;
  isActive: boolean;
  level: 1 | 2;
  onActive: () => void;
  row: NodeRow;
}) => {
  const { t } = useTranslation();

  return (
    <Box
      id={id}
      as="button"
      aria-level={level}
      aria-selected={isActive}
      bg={isActive ? 'bg.hover' : undefined}
      role="treeitem"
      tabIndex={-1}
      _hover={ROW_HOVER_PROPS}
      ps={level === 1 ? '1.5' : '5'}
      pe="1.5"
      py="1.5"
      rounded="md"
      textAlign="start"
      w="full"
      onClick={row.onAdd}
      onMouseEnter={onActive}
    >
      <HStack gap="2" justify="space-between" alignItems="start">
        <Stack gap="0" minW="0">
          <HStack gap="1.5" minW="0">
            {row.isBeta ? (
              <Tooltip content={t('widgets.workflow.addNodeDialog.betaNode')}>
                <Icon as={HammerIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
              </Tooltip>
            ) : null}
            <MiddleTruncate fontSize="md" fontWeight="600" text={row.title} />
          </HStack>
          {row.description ? (
            <Text color="fg.subtle" fontSize="xs" lineClamp={2} lineHeight="1.4">
              {row.description}
            </Text>
          ) : null}
        </Stack>
        <Badge variant="outline" fontFamily="mono">
          {row.nodePack}
        </Badge>
      </HStack>
    </Box>
  );
};

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
      bg={isActive ? 'bg.hover' : undefined}
      role="treeitem"
      tabIndex={-1}
      _hover={ROW_HOVER_PROPS}
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
        <Text flex="1" fontSize="md" fontWeight="700">
          {group.label}
        </Text>
        <Badge size="lg" variant="surface" fontFamily="mono">
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
  // The content stays mounted through the exit animation; a fresh mount per open resets search and expansion.
  const content = useExitPresence(isOpen);
  // The store clears the connection on close; keep filtering by it while the dialog animates out.
  const [shownConnectionFilter, setShownConnectionFilter] = useState(connectionFilter);

  if (isOpen && shownConnectionFilter !== connectionFilter) {
    setShownConnectionFilter(connectionFilter);
  }

  const onDialogOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onOpenChange(false);
      }
    },
    [onOpenChange]
  );

  return (
    <Dialog.Root
      lazyMount
      open={isOpen}
      scrollBehavior="inside"
      size="md"
      unmountOnExit
      onExitComplete={content.release}
      onOpenChange={onDialogOpenChange}
    >
      {content.isMounted ? (
        <AddNodeDialogContent
          key={content.generation}
          connectionFilter={shownConnectionFilter}
          onAddCurrentImage={onAddCurrentImage}
          onAddConnector={onAddConnector}
          onAddNode={onAddNode}
          onAddNote={onAddNote}
          onOpenChange={onOpenChange}
        />
      ) : null}
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
  const { t } = useTranslation();
  const { getProjectGraph } = useWorkflowUi();
  const error = useInvocationTemplatesSelector((snapshot) => snapshot.error);
  const status = useInvocationTemplatesSelector((snapshot) => snapshot.status);
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(() => new Set());
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(null);
  const [isRetryRequested, setIsRetryRequested] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const retryButtonRef = useRef<HTMLButtonElement | null>(null);
  const groupByCategory = useWorkflowPreferencesSelector((preferences) => preferences.workflowGroupNodesByCategory);
  const isSearching = searchTerm.trim().length > 0;

  // Each open mounts fresh content, so the search needs no reset here; clearing it would reflow the closing list.
  const close = useCallback(() => onOpenChange(false), [onOpenChange]);

  const groups = useMemo<CategoryGroup[]>(() => {
    const query = parseSearchQuery(searchTerm);
    const projectGraph = getProjectGraph();
    const shouldPromoteForReturn = isForIterationOutputConnection(
      connectionFilter,
      projectGraph.nodes,
      projectGraph.edges,
      templates
    );
    const utilityRows: NodeRow[] = [
      {
        description: t('widgets.workflow.addNodeDialog.connectorDescription'),
        isBeta: false,
        key: 'utility:connector',
        nodePack: 'invokeai',
        onAdd: () => {
          onAddConnector();
          close();
        },
        title: t('widgets.workflow.addNodeDialog.connectorTitle'),
      },
      ...(connectionFilter
        ? []
        : [
            {
              description: t('widgets.workflow.addNodeDialog.notesDescription'),
              isBeta: false,
              key: 'utility:notes',
              nodePack: 'invokeai',
              onAdd: () => {
                onAddNote();
                close();
              },
              title: t('widgets.workflow.addNodeDialog.notesTitle'),
            },
            {
              description: t('widgets.workflow.addNodeDialog.currentImageDescription'),
              isBeta: false,
              key: 'utility:current_image',
              nodePack: 'invokeai',
              onAdd: () => {
                onAddCurrentImage();
                close();
              },
              title: t('widgets.workflow.addNodeDialog.currentImageTitle'),
            },
          ]),
    ].flatMap((row) => {
      const rank = getSearchRank(query, row.title, '');

      return rank === null ? [] : [{ ...row, isUtility: true, rank }];
    });

    const byCategory = new Map<string, NodeRow[]>();

    for (const template of Object.values(templates)) {
      if (template.classification === 'internal' || !isCompatibleConnectionTemplate(template, connectionFilter)) {
        continue;
      }

      const rank = getTemplateSearchRank(template, query);

      if (rank === null) {
        continue;
      }

      const label = toCategoryLabel(template.category || 'other');
      const row: NodeRow = {
        description: template.description,
        isBeta: template.classification === 'beta',
        isUtility: false,
        key: `template:${template.type}`,
        nodePack: template.nodePack,
        onAdd: () => {
          onAddNode(template);
          close();
        },
        rank,
        title: template.title,
      };

      byCategory.set(label, [...(byCategory.get(label) ?? []), row]);
    }

    const groupsByLabel = [...byCategory.entries()].map(([label, rows]) => ({
      isUtility: false,
      label,
      rows: rows.sort((a, b) => compareRows(a, b, shouldPromoteForReturn)),
    }));
    const allGroups =
      utilityRows.length > 0
        ? [
            { isUtility: true, label: t('widgets.workflow.addNodeDialog.utilityCategory'), rows: utilityRows },
            ...groupsByLabel,
          ]
        : groupsByLabel;

    if (!groupByCategory) {
      const rows = allGroups.flatMap((group) => group.rows).sort((a, b) => compareRows(a, b, shouldPromoteForReturn));

      return rows.length > 0 ? [{ isUtility: false, label: '', rows }] : [];
    }

    // Rows are sorted, so each group lists by its best row; Utility leads among equals.
    return allGroups.sort((a, b) => {
      const bestA = a.rows[0]!;
      const bestB = b.rows[0]!;

      if (shouldPromoteForReturn && (bestA.key === FOR_RETURN_KEY) !== (bestB.key === FOR_RETURN_KEY)) {
        return bestA.key === FOR_RETURN_KEY ? -1 : 1;
      }

      if (bestA.rank !== bestB.rank) {
        return bestA.rank - bestB.rank;
      }

      if (a.isUtility !== b.isUtility) {
        return a.isUtility ? -1 : 1;
      }

      return a.label.localeCompare(b.label);
    });
  }, [
    close,
    connectionFilter,
    getProjectGraph,
    groupByCategory,
    onAddConnector,
    onAddCurrentImage,
    onAddNode,
    onAddNote,
    searchTerm,
    t,
    templates,
  ]);

  const totalCount = groups.reduce((sum, group) => sum + group.rows.length, 0);
  const isAllExpanded = groups.length > 0 && groups.every((group) => expandedCategories.has(group.label));

  const resultRows = useMemo<ResultRow[]>(() => {
    const rows: ResultRow[] = [];

    if (!groupByCategory) {
      return (groups[0]?.rows ?? []).map((row) => ({ id: row.key, kind: 'node', level: 1, row }));
    }

    for (const group of groups) {
      const isExpanded = isSearching || expandedCategories.has(group.label);
      rows.push({ group, id: `category:${group.label}`, isExpanded, kind: 'category' });

      if (isExpanded) {
        for (const row of group.rows) {
          rows.push({ id: row.key, kind: 'node', level: 2, row });
        }
      }
    }

    return rows;
  }, [expandedCategories, groupByCategory, groups, isSearching]);
  // While searching, Enter adds the best match rather than toggling the category header above it.
  const defaultActiveIndex = isSearching
    ? Math.max(
        0,
        resultRows.findIndex((row) => row.kind === 'node')
      )
    : 0;
  const effectiveActiveIndex =
    resultRows.length === 0
      ? null
      : activeIndex === null
        ? defaultActiveIndex
        : Math.min(activeIndex, resultRows.length - 1);
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

  const moveActiveIndex = useCallback(
    (direction: 1 | -1) => {
      setActiveIndex((prev) => {
        if (resultRows.length === 0) {
          return null;
        }

        const nextIndex =
          prev === null
            ? direction > 0
              ? Math.min(defaultActiveIndex + 1, resultRows.length - 1)
              : resultRows.length - 1
            : (prev + direction + resultRows.length) % resultRows.length;

        virtualizer.scrollToIndex(nextIndex, { align: 'auto' });
        return nextIndex;
      });
    },
    [defaultActiveIndex, resultRows.length, virtualizer]
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
      if (isImeComposing(event.nativeEvent)) {
        return;
      }

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
  const onSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.currentTarget.value);
    setActiveIndex(null);
  }, []);

  // The failure stays on screen, with its Retry busy, until the retried load settles; a later reload started
  // elsewhere shows as plain loading.
  if (isRetryRequested && status !== 'loading') {
    setIsRetryRequested(false);
  }

  const isRetrying = isRetryRequested && status === 'loading';
  const onRetryClick = useCallback(() => {
    if (isRetrying) {
      return;
    }

    setIsRetryRequested(true);
    ensureInvocationTemplatesLoaded();
  }, [isRetrying]);
  // A successful retry unmounts the focused Retry; the search, where adding a node starts, takes focus instead.
  const attachRetryButton = useCallback((node: HTMLButtonElement | null) => {
    const previous = retryButtonRef.current;

    retryButtonRef.current = node;

    if (node || !previous?.contains(document.activeElement)) {
      return;
    }

    queueMicrotask(() => {
      if (document.activeElement === null || document.activeElement === document.body) {
        searchRef.current?.focus();
      }
    });
  }, []);

  const connectionTypeLabel = connectionFilter ? getConnectionFilterTypeLabel(connectionFilter) : null;
  // Status messages sit beside the result tree rather than inside it: a tree may only own tree items.
  let statusMessage: ReactNode = null;
  let body: ReactNode = null;

  if (status === 'error' || isRetrying) {
    statusMessage = (
      <Stack alignItems="start" aria-busy={isRetrying || undefined} gap="2" px="1" py="4" role="alert">
        <Text color="fg.error" fontSize="md">
          {t('widgets.workflow.addNodeDialog.loadFailed')}
        </Text>
        {error ? (
          <Text color="fg.subtle" fontSize="xs">
            {error}
          </Text>
        ) : null}
        {/* aria-disabled rather than disabled while busy, so the Retry keeps focus. */}
        <Button
          ref={attachRetryButton}
          aria-busy={isRetrying || undefined}
          aria-disabled={isRetrying || undefined}
          size="sm"
          variant="outline"
          onClick={onRetryClick}
        >
          {isRetrying ? <Spinner boxSize="3" /> : null}
          {t('common.retry')}
        </Button>
      </Stack>
    );
  } else if (status !== 'loaded') {
    statusMessage = (
      <Text color="fg.subtle" fontSize="md" px="1" py="4">
        {t('widgets.workflow.addNodeDialog.loading')}
      </Text>
    );
  } else if (totalCount === 0) {
    statusMessage = (
      <Text color="fg.subtle" fontSize="md" px="1" py="4" textAlign="center">
        {!connectionFilter
          ? t('widgets.workflow.addNodeDialog.noMatches')
          : connectionTypeLabel === null
            ? t('widgets.workflow.addNodeDialog.noCompatibleMatchesAnyType')
            : t('widgets.workflow.addNodeDialog.noCompatibleMatches', { type: connectionTypeLabel })}
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
                  ref={searchRef}
                  autoFocus
                  aria-activedescendant={
                    effectiveActiveIndex === null ? undefined : getResultRowId(effectiveActiveIndex)
                  }
                  aria-controls={RESULT_LIST_ID}
                  aria-expanded="true"
                  aria-label={t('widgets.workflow.addNodeDialog.search')}
                  flex="1"
                  placeholder={
                    !connectionFilter
                      ? t('widgets.workflow.addNodeDialog.searchPlaceholder')
                      : connectionTypeLabel === null
                        ? t('widgets.workflow.addNodeDialog.searchCompatiblePlaceholderAnyType')
                        : t('widgets.workflow.addNodeDialog.searchCompatiblePlaceholder', { type: connectionTypeLabel })
                  }
                  role="combobox"
                  size="lg"
                  value={searchTerm}
                  onChange={onSearchChange}
                  onKeyDown={onSearchKeyDown}
                />
                {groupByCategory ? (
                  <Tooltip
                    content={
                      isAllExpanded
                        ? t('widgets.workflow.addNodeDialog.collapseAll')
                        : t('widgets.workflow.addNodeDialog.expandAll')
                    }
                  >
                    <IconButton
                      aria-label={
                        isAllExpanded
                          ? t('widgets.workflow.addNodeDialog.collapseAllCategories')
                          : t('widgets.workflow.addNodeDialog.expandAllCategories')
                      }
                      size="lg"
                      variant="ghost"
                      onClick={toggleAllCategories}
                    >
                      <Icon as={isAllExpanded ? ChevronsUpDownIcon : ChevronsDownUpIcon} />
                    </IconButton>
                  </Tooltip>
                ) : null}
              </HStack>
              <ScrollArea.Root flex="1" minH="0" variant="hover" w="full">
                <ScrollArea.Viewport ref={setScrollElement} h="full" w="full">
                  <ScrollArea.Content w="full">
                    {statusMessage}
                    <Box
                      id={RESULT_LIST_ID}
                      aria-label={t('widgets.workflow.addNodeDialog.results')}
                      role="tree"
                      w="full"
                    >
                      {body}
                    </Box>
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
        <NodeResultRow id={id} isActive={isActive} level={row.level} onActive={onActive} row={row.row} />
      )}
    </Box>
  );
};
