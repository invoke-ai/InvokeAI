/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { NodePackInfo } from '@features/nodes/core/catalog';

import { Badge, Flex, Icon, Input, InputGroup, Spinner, Stack, Text } from '@chakra-ui/react';
import { filterNodePacks, isProblemPack, type NodePackFilters } from '@features/nodes/core/library';
import { refreshCustomNodePacks } from '@features/nodes/data/nodesStore';
import { openNodesManagerTab } from '@features/nodes/ui/nodesUiStore';
import { Button, Row, Scrollable, Tooltip } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { ArrowRightIcon, BlocksIcon, PackageOpenIcon, SearchIcon, TriangleAlertIcon } from 'lucide-react';
import { useDeferredValue, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { NodePackContextMenu, type NodePackContextMenuTarget } from './NodePackContextMenu';
import { NodePackFilterMenu } from './NodePackFilterMenu';

/**
 * Master list for the nodes manager: a search box over a scrollable column of
 * selectable pack rows. Mirrors the model library list — `Row` with the
 * `accent` active variant marks the open pack, and the search filters by name
 * or path.
 */
export const NodePackList = ({
  activePackName,
  error,
  filters,
  onFiltersChange,
  onSelect,
  onUninstalled,
  packs,
  status,
}: {
  activePackName: string | null;
  error: string | null;
  filters: NodePackFilters;
  onFiltersChange: (next: NodePackFilters) => void;
  onSelect: (packName: string) => void;
  onUninstalled: (packName: string) => void;
  packs: NodePackInfo[];
  status: 'idle' | 'loading' | 'loaded' | 'error';
}) => {
  const { t } = useTranslation();
  const [contextMenuTarget, setContextMenuTarget] = useState<NodePackContextMenuTarget | null>(null);
  const deferredFilters = useDeferredValue(filters);
  const filtered = useMemo(() => filterNodePacks(packs, deferredFilters), [deferredFilters, packs]);

  if (status === 'error') {
    return (
      <Flex align="center" flex="1" justify="center" minH="0" p="3">
        <EmptyState
          danger
          description={error}
          icon={<Icon as={TriangleAlertIcon} />}
          title={t('nodes.couldNotLoadPacks')}
        >
          <Button size="sm" variant="outline" onClick={() => void refreshCustomNodePacks()}>
            {t('common.retry')}
          </Button>
        </EmptyState>
      </Flex>
    );
  }

  return (
    <Stack flex="1" gap="2" minH="0" pt="3">
      <Flex gap="1.5" px="3">
        <InputGroup startElement={<Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />}>
          <Input
            aria-label={t('nodes.searchPacks')}
            placeholder={t('nodes.searchPacksPlaceholder')}
            size="xs"
            value={filters.searchTerm}
            onChange={(event) => onFiltersChange({ ...filters, searchTerm: event.currentTarget.value })}
          />
        </InputGroup>
        <NodePackFilterMenu filters={filters} onChange={onFiltersChange} />
      </Flex>
      <Scrollable h="full" label={t('nodes.installedPacks')} minH="0">
        {status === 'idle' || status === 'loading' ? (
          <Flex align="center" justify="center" py="10">
            <Spinner color="fg.subtle" size="sm" />
          </Flex>
        ) : packs.length === 0 ? (
          <EmptyState
            description={t('nodes.noPacksDescription')}
            icon={<Icon as={PackageOpenIcon} />}
            title={t('nodes.noPacks')}
          >
            <Button size="sm" onClick={() => openNodesManagerTab('add')}>
              {t('nodes.addNodes')}
              <Icon as={ArrowRightIcon} />
            </Button>
          </EmptyState>
        ) : filtered.length === 0 ? (
          <EmptyState
            description={t('nodes.tryDifferentSearch')}
            icon={<Icon as={SearchIcon} />}
            title={t('nodes.noPacksMatch')}
          >
            <Button size="sm" variant="outline" onClick={() => openNodesManagerTab('add')}>
              {t('nodes.addNodes')}
              <Icon as={ArrowRightIcon} />
            </Button>
          </EmptyState>
        ) : (
          <Stack gap="1" minW="0" p="1" px="3" w="full">
            {filtered.map((pack) => (
              <PackRow
                key={pack.name}
                isActive={pack.name === activePackName}
                pack={pack}
                problemHint={t('nodes.noNodesRegisteredHint')}
                onContextMenu={(targetPack, x, y) => setContextMenuTarget({ pack: targetPack, x, y })}
                onSelect={() => onSelect(pack.name)}
              />
            ))}
          </Stack>
        )}
      </Scrollable>
      <NodePackContextMenu
        target={contextMenuTarget}
        onClose={() => setContextMenuTarget(null)}
        onUninstalled={onUninstalled}
      />
    </Stack>
  );
};

const PackRow = ({
  isActive,
  onContextMenu,
  onSelect,
  problemHint,
  pack,
}: {
  isActive: boolean;
  onContextMenu: (pack: NodePackInfo, x: number, y: number) => void;
  onSelect: () => void;
  /** Tooltip for the zero-node warning badge. */
  problemHint: string;
  pack: NodePackInfo;
}) => (
  <Row
    active={isActive ? 'accent' : 'none'}
    aria-current={isActive || undefined}
    px="2"
    py="1.5"
    minW="0"
    overflow="hidden"
    role="button"
    rounded="md"
    tabIndex={0}
    onClick={onSelect}
    onContextMenu={(event) => {
      event.preventDefault();
      onContextMenu(pack, event.clientX, event.clientY);
    }}
    onKeyDown={(event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSelect();
      }
    }}
  >
    <Icon as={BlocksIcon} boxSize="4" color={isActive ? 'accent.contrast' : 'fg.subtle'} flexShrink={0} />
    <Text fontSize="xs" fontWeight="600" maxW="full" truncate>
      {pack.name}
    </Text>
    {isProblemPack(pack) ? (
      // Zero nodes is the strongest health signal the catalog carries: the
      // pack's import failed or a reload/restart is pending.
      <Tooltip content={problemHint}>
        <Badge colorPalette="orange" flexShrink={0} fontSize="2xs" ms="auto" variant="surface">
          {pack.nodeCount}
        </Badge>
      </Tooltip>
    ) : (
      <Badge
        colorPalette={isActive ? undefined : 'gray'}
        flexShrink={0}
        fontSize="2xs"
        variant={isActive ? 'solid' : 'surface'}
        ms="auto"
      >
        {pack.nodeCount}
      </Badge>
    )}
  </Row>
);
