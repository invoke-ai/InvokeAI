import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Box,
  Flex,
  Icon,
  Input,
  Modal,
  ModalBody,
  ModalContent,
  ModalOverlay,
  Spacer,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { CommandEmpty, CommandItem, CommandList, CommandRoot } from 'cmdk';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useBuildNode } from 'features/nodes/hooks/useBuildNode';
import {
  $cursorPos,
  $edgePendingUpdate,
  $pendingConnection,
  $templates,
  edgesChanged,
  nodesChanged,
  useAddNodeCmdk,
} from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { findUnoccupiedPosition } from 'features/nodes/store/util/findUnoccupiedPosition';
import { getFirstValidConnection } from 'features/nodes/store/util/getFirstValidConnection';
import { connectionToEdge } from 'features/nodes/store/util/reactFlowUtil';
import { validateConnectionTypes } from 'features/nodes/store/util/validateConnectionTypes';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memoize } from 'lodash-es';
import { computed } from 'nanostores';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCircuitryBold, PiFlaskBold, PiHammerBold } from 'react-icons/pi';
import type { EdgeChange, NodeChange } from 'reactflow';
import type { S } from 'services/api/types';

const useThrottle = <T,>(value: T, limit: number) => {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastRan = useRef(Date.now());

  useEffect(() => {
    const handler = setTimeout(
      function () {
        if (Date.now() - lastRan.current >= limit) {
          setThrottledValue(value);
          lastRan.current = Date.now();
        }
      },
      limit - (Date.now() - lastRan.current)
    );

    return () => {
      clearTimeout(handler);
    };
  }, [value, limit]);

  return throttledValue;
};

const useAddNode = () => {
  const { t } = useTranslation();
  const store = useAppStore();
  const buildInvocation = useBuildNode();
  const templates = useStore($templates);
  const pendingConnection = useStore($pendingConnection);

  const addNode = useCallback(
    (nodeType: string): void => {
      const node = buildInvocation(nodeType);
      if (!node) {
        const errorMessage = t('nodes.unknownNode', {
          nodeType: nodeType,
        });
        toast({
          status: 'error',
          title: errorMessage,
        });
        return;
      }

      // Find a cozy spot for the node
      const cursorPos = $cursorPos.get();
      const { nodes, edges } = selectNodesSlice(store.getState());
      node.position = findUnoccupiedPosition(nodes, cursorPos?.x ?? node.position.x, cursorPos?.y ?? node.position.y);
      node.selected = true;

      // Deselect all other nodes and edges
      const nodeChanges: NodeChange[] = [{ type: 'add', item: node }];
      const edgeChanges: EdgeChange[] = [];
      nodes.forEach(({ id, selected }) => {
        if (selected) {
          nodeChanges.push({ type: 'select', id, selected: false });
        }
      });
      edges.forEach(({ id, selected }) => {
        if (selected) {
          edgeChanges.push({ type: 'select', id, selected: false });
        }
      });

      // Onwards!
      if (nodeChanges.length > 0) {
        store.dispatch(nodesChanged(nodeChanges));
      }
      if (edgeChanges.length > 0) {
        store.dispatch(edgesChanged(edgeChanges));
      }

      // Auto-connect an edge if we just added a node and have a pending connection
      if (pendingConnection && isInvocationNode(node)) {
        const edgePendingUpdate = $edgePendingUpdate.get();
        const { handleType } = pendingConnection;

        const source = handleType === 'source' ? pendingConnection.nodeId : node.id;
        const sourceHandle = handleType === 'source' ? pendingConnection.handleId : null;
        const target = handleType === 'target' ? pendingConnection.nodeId : node.id;
        const targetHandle = handleType === 'target' ? pendingConnection.handleId : null;

        const { nodes, edges } = selectNodesSlice(store.getState());
        const connection = getFirstValidConnection(
          source,
          sourceHandle,
          target,
          targetHandle,
          nodes,
          edges,
          templates,
          edgePendingUpdate
        );
        if (connection) {
          const newEdge = connectionToEdge(connection);
          store.dispatch(edgesChanged([{ type: 'add', item: newEdge }]));
        }
      }
    },
    [buildInvocation, pendingConnection, store, t, templates]
  );

  return addNode;
};

const cmdkRootSx: SystemStyleObject = {
  '[cmdk-root]': {
    w: 'full',
    h: 'full',
  },
  '[cmdk-list]': {
    w: 'full',
    h: 'full',
  },
};

export const AddNodeCmdk = memo(() => {
  const { t } = useTranslation();
  const addNodeCmdk = useAddNodeCmdk();
  const inputRef = useRef<HTMLInputElement>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const addNode = useAddNode();
  const tab = useAppSelector(selectActiveTab);
  const throttledSearchTerm = useThrottle(searchTerm, 100);

  useRegisteredHotkeys({
    id: 'addNode',
    category: 'workflows',
    callback: addNodeCmdk.setTrue,
    options: { enabled: tab === 'workflows', preventDefault: true },
    dependencies: [addNodeCmdk.setTrue, tab],
  });

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  const onClose = useCallback(() => {
    addNodeCmdk.setFalse();
    setSearchTerm('');
    $pendingConnection.set(null);
  }, [addNodeCmdk]);

  const onSelect = useCallback(
    (value: string) => {
      addNode(value);
      onClose();
    },
    [addNode, onClose]
  );

  return (
    <Modal
      isOpen={addNodeCmdk.isTrue}
      onClose={onClose}
      useInert={false}
      initialFocusRef={inputRef}
      size="xl"
      isCentered
    >
      <ModalOverlay />
      <ModalContent h="512" maxH="70%">
        <ModalBody p={2} h="full" sx={cmdkRootSx}>
          <CommandRoot loop shouldFilter={false}>
            <Flex flexDir="column" h="full" gap={2}>
              <Input ref={inputRef} value={searchTerm} onChange={onChange} placeholder={t('nodes.nodeSearch')} />
              <Box w="full" h="full">
                <ScrollableContent>
                  <CommandEmpty>
                    <IAINoContentFallback
                      position="absolute"
                      top={0}
                      right={0}
                      bottom={0}
                      left={0}
                      icon={null}
                      label="No matching items"
                    />
                  </CommandEmpty>
                  <CommandList>
                    <NodeCommandList searchTerm={throttledSearchTerm} onSelect={onSelect} />
                  </CommandList>
                </ScrollableContent>
              </Box>
            </Flex>
          </CommandRoot>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

AddNodeCmdk.displayName = 'AddNodeCmdk';

const cmdkItemSx: SystemStyleObject = {
  '&[data-selected="true"]': {
    bg: 'base.700',
  },
};

type NodeCommandItemData = {
  value: string;
  label: string;
  description: string;
  classification: S['Classification'];
  nodePack: string;
};

/**
 * An array of all templates, excluding deprecated ones.
 */
const $templatesArray = computed($templates, (templates) =>
  Object.values(templates).filter((template) => template.classification !== 'deprecated')
);

const createRegex = memoize(
  (inputValue: string) =>
    new RegExp(
      inputValue
        .trim()
        .replace(/[-[\]{}()*+!<=:?./\\^$|#,]/g, '')
        .split(' ')
        .join('.*'),
      'gi'
    )
);

// Filterable items are a subset of Invocation template - we also want to filter for notes or current image node,
// so we are using a less specific type instead of `InvocationTemplate`
type FilterableItem = {
  type: string;
  title: string;
  description: string;
  tags: string[];
  classification: S['Classification'];
  nodePack: string;
};

const filter = memoize(
  (item: FilterableItem, searchTerm: string) => {
    const regex = createRegex(searchTerm);

    if (!searchTerm) {
      return true;
    }

    if (item.title.includes(searchTerm) || regex.test(item.title)) {
      return true;
    }

    if (item.type.includes(searchTerm) || regex.test(item.type)) {
      return true;
    }

    if (item.description.includes(searchTerm) || regex.test(item.description)) {
      return true;
    }

    if (item.nodePack.includes(searchTerm) || regex.test(item.nodePack)) {
      return true;
    }

    if (item.classification.includes(searchTerm) || regex.test(item.classification)) {
      return true;
    }

    for (const tag of item.tags) {
      if (tag.includes(searchTerm) || regex.test(tag)) {
        return true;
      }
    }

    return false;
  },
  (item: FilterableItem, searchTerm: string) => `${item.type}-${searchTerm}`
);

const NodeCommandList = memo(({ searchTerm, onSelect }: { searchTerm: string; onSelect: (value: string) => void }) => {
  const { t } = useTranslation();
  const templatesArray = useStore($templatesArray);
  const pendingConnection = useStore($pendingConnection);
  const currentImageFilterItem = useMemo<FilterableItem>(
    () => ({
      type: 'current_image',
      title: t('nodes.currentImage'),
      description: t('nodes.currentImageDescription'),
      tags: ['progress', 'image', 'current'],
      classification: 'stable',
      nodePack: 'invokeai',
    }),
    [t]
  );
  const notesFilterItem = useMemo<FilterableItem>(
    () => ({
      type: 'notes',
      title: t('nodes.notes'),
      description: t('nodes.notesDescription'),
      tags: ['notes'],
      classification: 'stable',
      nodePack: 'invokeai',
    }),
    [t]
  );

  const items = useMemo<NodeCommandItemData[]>(() => {
    // If we have a connection in progress, we need to filter the node choices
    const _items: NodeCommandItemData[] = [];

    if (!pendingConnection) {
      for (const template of templatesArray) {
        if (filter(template, searchTerm)) {
          _items.push({
            label: template.title,
            value: template.type,
            description: template.description,
            classification: template.classification,
            nodePack: template.nodePack,
          });
        }
      }

      for (const item of [currentImageFilterItem, notesFilterItem]) {
        if (filter(item, searchTerm)) {
          _items.push({
            label: item.title,
            value: item.type,
            description: item.description,
            classification: item.classification,
            nodePack: item.nodePack,
          });
        }
      }
    } else {
      for (const template of templatesArray) {
        if (filter(template, searchTerm)) {
          const candidateFields = pendingConnection.handleType === 'source' ? template.inputs : template.outputs;

          for (const field of Object.values(candidateFields)) {
            const sourceType =
              pendingConnection.handleType === 'source' ? field.type : pendingConnection.fieldTemplate.type;
            const targetType =
              pendingConnection.handleType === 'target' ? field.type : pendingConnection.fieldTemplate.type;

            if (validateConnectionTypes(sourceType, targetType)) {
              _items.push({
                label: template.title,
                value: template.type,
                description: template.description,
                classification: template.classification,
                nodePack: template.nodePack,
              });
              break;
            }
          }
        }
      }
    }

    return _items;
  }, [pendingConnection, currentImageFilterItem, searchTerm, notesFilterItem, templatesArray]);

  return (
    <>
      {items.map((item) => (
        <CommandItem key={item.value} value={item.value} onSelect={onSelect} asChild>
          <Flex role="button" flexDir="column" sx={cmdkItemSx} py={1} px={2} borderRadius="base">
            <Flex alignItems="center" gap={2}>
              {item.classification === 'beta' && <Icon boxSize={4} color="invokeYellow.300" as={PiHammerBold} />}
              {item.classification === 'prototype' && <Icon boxSize={4} color="invokeRed.300" as={PiFlaskBold} />}
              {item.classification === 'internal' && <Icon boxSize={4} color="invokePurple.300" as={PiCircuitryBold} />}
              <Text fontWeight="semibold">{item.label}</Text>
              <Spacer />
              <Text variant="subtext" fontWeight="semibold">
                {item.nodePack}
              </Text>
            </Flex>
            {item.description && <Text color="base.200">{item.description}</Text>}
          </Flex>
        </CommandItem>
      ))}
    </>
  );
});

NodeCommandList.displayName = 'CommandListItems';
