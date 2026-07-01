/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { NodePackInfo } from '@workbench/customNodes/api';

import { Badge, Flex, Icon, Input, InputGroup, Spinner, Stack, Text } from '@chakra-ui/react';
import { Button, Row, Scrollable } from '@workbench/components/ui';
import { EmptyState } from '@workbench/components/ui/EmptyState';
import { refreshCustomNodePacks } from '@workbench/customNodes/nodesStore';
import { BlocksIcon, PackageOpenIcon, SearchIcon, TriangleAlertIcon } from 'lucide-react';
import { useDeferredValue, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { NodePackContextMenu, type NodePackContextMenuTarget } from './NodePackContextMenu';

/**
 * Master list for the nodes manager: a search box over a scrollable column of
 * selectable pack rows. Mirrors the model library list — `Row` with the
 * `accent` active variant marks the open pack, and the search filters by name
 * or path.
 */
export const NodePackList = ({
  activePackName,
  error,
  onSearchChange,
  onSelect,
  onUninstalled,
  packs,
  searchTerm,
  status,
}: {
  activePackName: string | null;
  error: string | null;
  onSearchChange: (value: string) => void;
  onSelect: (packName: string) => void;
  onUninstalled: (packName: string) => void;
  packs: NodePackInfo[];
  searchTerm: string;
  status: 'idle' | 'loading' | 'loaded' | 'error';
}) => {
  const { t } = useTranslation();
  const [contextMenuTarget, setContextMenuTarget] = useState<NodePackContextMenuTarget | null>(null);
  const deferredSearchTerm = useDeferredValue(searchTerm);
  const filtered = useMemo(() => {
    const query = deferredSearchTerm.trim().toLowerCase();

    if (!query) {
      return packs;
    }

    return packs.filter((pack) => pack.name.toLowerCase().includes(query) || pack.path.toLowerCase().includes(query));
  }, [deferredSearchTerm, packs]);

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
      <InputGroup px="3" startElement={<Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />}>
        <Input
          aria-label={t('nodes.searchPacks')}
          placeholder={t('nodes.searchPacksPlaceholder')}
          size="xs"
          value={searchTerm}
          onChange={(event) => onSearchChange(event.currentTarget.value)}
        />
      </InputGroup>
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
          />
        ) : filtered.length === 0 ? (
          <EmptyState
            description={t('nodes.tryDifferentSearch')}
            icon={<Icon as={SearchIcon} />}
            title={t('nodes.noPacksMatch')}
          />
        ) : (
          <Stack gap="1" minW="0" p="1" px="3" w="full">
            {filtered.map((pack) => (
              <PackRow
                key={pack.name}
                isActive={pack.name === activePackName}
                pack={pack}
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
  pack,
}: {
  isActive: boolean;
  onContextMenu: (pack: NodePackInfo, x: number, y: number) => void;
  onSelect: () => void;
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
    <Badge
      colorPalette={isActive ? undefined : 'gray'}
      flexShrink={0}
      fontSize="2xs"
      variant={isActive ? 'solid' : 'surface'}
      ms="auto"
    >
      {pack.node_count}
    </Badge>
  </Row>
);
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
