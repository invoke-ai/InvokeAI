import type { AddNodeConnectionFilter } from '@workbench/widgets/workflow/workflowUiStore';
import type { InvocationTemplate } from '@workbench/workflows/types';

import { Accordion, Badge, Box, Dialog, HStack, Icon, Input, Portal, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui/Button';
import { Scrollable } from '@workbench/components/ui/Scrollable';
import { Tooltip } from '@workbench/components/ui/Tooltip';
import { useInvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import { getCompatibleInputTemplate } from '@workbench/workflows/validation';
import { HammerIcon } from 'lucide-react';
import { useCallback, useMemo, useState, type ChangeEvent, type ReactNode } from 'react';

/**
 * Command-palette-style node picker: a centered search dialog whose results
 * are grouped into collapsible categories (with counts), mirroring the legacy
 * editor's add-node menu. Beta nodes carry the hammer icon. The two UI-only
 * nodes (Notes, Current Image) lead as a "Utility" group. No result cap —
 * searching auto-expands every group; idle shows all categories collapsed.
 */

const UTILITY_CATEGORY = 'Utility';

const toCategoryLabel = (value: string): string =>
  value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .trim();

const matchesSearch = (template: InvocationTemplate, terms: string[]): boolean => {
  const haystack = `${template.title} ${template.type} ${template.tags.join(' ')} ${template.category}`.toLowerCase();

  return terms.every((term) => haystack.includes(term));
};

interface NodeRow {
  description: string;
  isBeta: boolean;
  nodePack: string;
  onAdd: () => void;
  title: string;
}

interface CategoryGroup {
  label: string;
  rows: NodeRow[];
}

const NodeResultRow = ({ row }: { row: NodeRow }) => (
  <Box
    as="button"
    _hover={{ bg: 'bg.emphasized' }}
    ps="3"
    pe="2"
    py="2"
    rounded="md"
    textAlign="start"
    w="full"
    onClick={row.onAdd}
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
          <Text fontSize="sm" fontWeight="600" truncate>
            {row.title}
          </Text>
        </HStack>
        {row.description ? (
          <Text color="fg.subtle" fontSize="2xs" lineClamp={2}>
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

export const AddNodeDialog = ({
  connectionFilter,
  isOpen,
  onAddCurrentImage,
  onAddNode,
  onAddNote,
  onOpenChange,
}: {
  connectionFilter: AddNodeConnectionFilter | null;
  isOpen: boolean;
  onAddCurrentImage: () => void;
  onAddNode: (template: InvocationTemplate) => void;
  onAddNote: () => void;
  onOpenChange: (isOpen: boolean) => void;
}) =>
  isOpen ? (
    <AddNodeDialogContent
      connectionFilter={connectionFilter}
      isOpen={isOpen}
      onAddCurrentImage={onAddCurrentImage}
      onAddNode={onAddNode}
      onAddNote={onAddNote}
      onOpenChange={onOpenChange}
    />
  ) : null;

const AddNodeDialogContent = ({
  connectionFilter,
  isOpen,
  onAddCurrentImage,
  onAddNode,
  onAddNote,
  onOpenChange,
}: {
  connectionFilter: AddNodeConnectionFilter | null;
  isOpen: boolean;
  onAddCurrentImage: () => void;
  onAddNode: (template: InvocationTemplate) => void;
  onAddNote: () => void;
  onOpenChange: (isOpen: boolean) => void;
}) => {
  const { error, status, templates } = useInvocationTemplatesSnapshot();
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<string[]>([]);
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
    const utilityRows: NodeRow[] = connectionFilter
      ? []
      : [
          {
            description: 'Annotate the workflow with a free-text note.',
            isBeta: false,
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
            nodePack: 'invokeai',
            onAdd: () => {
              onAddCurrentImage();
              close(true);
            },
            title: 'Current Image',
          },
        ].filter((row) => terms.every((term) => row.title.toLowerCase().includes(term)));

    const byCategory = new Map<string, NodeRow[]>();

    for (const template of Object.values(templates)) {
      if (
        template.classification === 'internal' ||
        (terms.length > 0 && !matchesSearch(template, terms)) ||
        (connectionFilter && !getCompatibleInputTemplate(template, connectionFilter.sourceType))
      ) {
        continue;
      }

      const label = toCategoryLabel(template.category || 'other');
      const row: NodeRow = {
        description: template.description,
        isBeta: template.classification === 'beta',
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
  }, [close, connectionFilter, onAddCurrentImage, onAddNode, onAddNote, searchTerm, templates]);

  const totalCount = groups.reduce((sum, group) => sum + group.rows.length, 0);
  const accordionValue = isSearching ? groups.map((group) => group.label) : expandedCategories;

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
        {connectionFilter ? `No nodes accept ${connectionFilter.sourceType.name}.` : 'No nodes match your search.'}
      </Text>
    );
  } else {
    body = (
      <Accordion.Root
        multiple
        value={accordionValue}
        variant="plain"
        onValueChange={(event) => setExpandedCategories(event.value)}
      >
        {groups.map((group) => {
          const isExpanded = accordionValue.includes(group.label);

          return (
            <Accordion.Item key={group.label} value={group.label} _hover={{ bg: 'bg.muted' }} rounded="md">
              <Accordion.ItemTrigger cursor="pointer" ps="1" pe="2" py="1.5">
                <Accordion.ItemIndicator />
                <Text flex="1" fontSize="xs" fontWeight="700" textAlign="start">
                  {group.label}
                </Text>
                <Badge size="sm" variant="surface" fontFamily="mono">
                  {group.rows.length}
                </Badge>
              </Accordion.ItemTrigger>
              <Accordion.ItemContent>
                {isExpanded ? (
                  <Stack gap="0">
                    {group.rows.map((row) => (
                      <NodeResultRow key={`${group.label}:${row.title}`} row={row} />
                    ))}
                  </Stack>
                ) : null}
              </Accordion.ItemContent>
            </Accordion.Item>
          );
        })}
      </Accordion.Root>
    );
  }

  return (
    <Dialog.Root
      open={isOpen}
      placement="top"
      scrollBehavior="inside"
      size="lg"
      onOpenChange={(event) => {
        if (!event.open) {
          close(false);
        }
      }}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" color="fg" mt="16">
            <Dialog.Body p="3">
              <Stack gap="2">
                <Input
                  autoFocus
                  aria-label="Search for nodes"
                  placeholder={
                    connectionFilter
                      ? `Search compatible ${connectionFilter.sourceType.name} nodes…`
                      : 'Search for nodes…'
                  }
                  size="md"
                  value={searchTerm}
                  onChange={(event: ChangeEvent<HTMLInputElement>) => setSearchTerm(event.currentTarget.value)}
                />
                <HStack gap="2">
                  <Button
                    disabled={isSearching}
                    size="2xs"
                    variant="ghost"
                    onClick={() => setExpandedCategories(groups.map((group) => group.label))}
                  >
                    Expand All
                  </Button>
                  <Button disabled={isSearching} size="2xs" variant="ghost" onClick={() => setExpandedCategories([])}>
                    Collapse All
                  </Button>
                </HStack>
                <Scrollable label="Node search results" maxH="60vh" minH="12rem">
                  {body}
                </Scrollable>
              </Stack>
            </Dialog.Body>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
