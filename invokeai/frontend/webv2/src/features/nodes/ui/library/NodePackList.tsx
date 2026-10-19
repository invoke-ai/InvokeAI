/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { NodePackInfo } from '@features/nodes/core/catalog';
import type { ListRowProps } from '@platform/ui/list/List';
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';

import { Badge, Flex, Icon, Input, InputGroup, Stack } from '@chakra-ui/react';
import { filterNodePacks, isProblemPack, type NodePackFilters } from '@features/nodes/core/library';
import { refreshCustomNodePacks } from '@features/nodes/data/nodesStore';
import { openNodesManagerTab } from '@features/nodes/ui/nodesUiStore';
import { Button, Tooltip } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { listRowsFromItems } from '@platform/ui/list/listRows';
import { ArrowRightIcon, BlocksIcon, PackageOpenIcon, SearchIcon, TriangleAlertIcon } from 'lucide-react';
import { useCallback, useDeferredValue, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { NodePackContextMenu, type NodePackContextMenuTarget } from './NodePackContextMenu';
import { NodePackFilterMenu } from './NodePackFilterMenu';

const getPackName = (pack: NodePackInfo): string => pack.name;

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
  const rows = useMemo(
    () => listRowsFromItems(filterNodePacks(packs, deferredFilters), getPackName),
    [deferredFilters, packs]
  );
  const handleCloseContextMenu = useCallback(() => {
    contextMenuTarget?.restoreFocus();
    setContextMenuTarget(null);
  }, [contextMenuTarget]);

  const renderItem = (pack: NodePackInfo, rowProps: ListRowProps) => (
    <ListItem
      {...rowProps}
      isMenuOpen={contextMenuTarget?.pack.name === pack.name}
      leading={
        <Icon as={BlocksIcon} boxSize="4" color={rowProps.isActive ? 'accent.contrast' : 'fg.subtle'} flexShrink={0} />
      }
      title={pack.name}
      trailing={
        isProblemPack(pack) ? (
          // Zero registered nodes indicates import failure or pending reload/restart.
          <Tooltip content={t('nodes.noNodesRegisteredHint')}>
            <Badge colorPalette="orange" fontSize="xs" variant="surface">
              {pack.nodeCount}
            </Badge>
          </Tooltip>
        ) : (
          <Badge
            colorPalette={rowProps.isActive ? undefined : 'gray'}
            fontSize="xs"
            variant={rowProps.isActive ? 'solid' : 'surface'}
          >
            {pack.nodeCount}
          </Badge>
        )
      }
      onContextMenu={(anchor: ListContextMenuAnchor) => setContextMenuTarget({ ...anchor, pack })}
      onPress={() => onSelect(pack.name)}
    />
  );

  return (
    <Stack flex="1" gap="2" minH="0" pt="3">
      <Flex gap="1.5" px="3">
        <InputGroup startElement={<Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />}>
          <Input
            aria-label={t('nodes.searchPacks')}
            placeholder={t('nodes.searchPacksPlaceholder')}
            value={filters.searchTerm}
            onChange={(event) => onFiltersChange({ ...filters, searchTerm: event.currentTarget.value })}
          />
        </InputGroup>
        <NodePackFilterMenu filters={filters} onChange={onFiltersChange} />
      </Flex>
      <List
        activeKey={activePackName}
        density="compact"
        emptyState={
          packs.length === 0 ? (
            <EmptyState
              description={t('nodes.noPacksDescription')}
              icon={<Icon as={PackageOpenIcon} />}
              title={t('nodes.noPacks')}
            >
              <Button size="lg" onClick={() => openNodesManagerTab('add')}>
                {t('nodes.addNodes')}
                <Icon as={ArrowRightIcon} />
              </Button>
            </EmptyState>
          ) : (
            <EmptyState
              description={t('nodes.tryDifferentSearch')}
              icon={<Icon as={SearchIcon} />}
              title={t('nodes.noPacksMatch')}
            >
              <Button size="lg" variant="outline" onClick={() => openNodesManagerTab('add')}>
                {t('nodes.addNodes')}
                <Icon as={ArrowRightIcon} />
              </Button>
            </EmptyState>
          )
        }
        errorState={
          <EmptyState
            danger
            description={error}
            icon={<Icon as={TriangleAlertIcon} />}
            title={t('nodes.couldNotLoadPacks')}
          >
            <Button size="lg" variant="outline" onClick={() => void refreshCustomNodePacks()}>
              {t('common.retry')}
            </Button>
          </EmptyState>
        }
        label={t('nodes.installedPacks')}
        renderItem={renderItem}
        rows={rows}
        status={status === 'error' ? 'error' : status === 'loaded' ? 'ready' : 'loading'}
      />
      <NodePackContextMenu target={contextMenuTarget} onClose={handleCloseContextMenu} onUninstalled={onUninstalled} />
    </Stack>
  );
};
