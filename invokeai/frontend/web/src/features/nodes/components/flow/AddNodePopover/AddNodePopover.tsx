import 'reactflow/dist/style.css';

import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, Flex, Popover, PopoverAnchor, PopoverBody, PopoverContent } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppStore } from 'app/store/storeHooks';
import type { SelectInstance } from 'chakra-react-select';
import { INTERACTION_SCOPES } from 'common/hooks/interactionScopes';
import { useBuildNode } from 'features/nodes/hooks/useBuildNode';
import {
  $cursorPos,
  $edgePendingUpdate,
  $isAddNodePopoverOpen,
  $pendingConnection,
  $templates,
  closeAddNodePopover,
  edgesChanged,
  nodesChanged,
  openAddNodePopover,
} from 'features/nodes/store/nodesSlice';
import { findUnoccupiedPosition } from 'features/nodes/store/util/findUnoccupiedPosition';
import { getFirstValidConnection } from 'features/nodes/store/util/getFirstValidConnection';
import { connectionToEdge } from 'features/nodes/store/util/reactFlowUtil';
import { validateConnectionTypes } from 'features/nodes/store/util/validateConnectionTypes';
import type { AnyNode } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { toast } from 'features/toast/toast';
import { filter, map, memoize, some } from 'lodash-es';
import { memo, useCallback, useMemo, useRef } from 'react';
import { flushSync } from 'react-dom';
import { useHotkeys } from 'react-hotkeys-hook';
import type { HotkeyCallback } from 'react-hotkeys-hook/dist/types';
import { useTranslation } from 'react-i18next';
import type { FilterOptionOption } from 'react-select/dist/declarations/src/filters';
import type { EdgeChange, NodeChange } from 'reactflow';

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

const filterOption = memoize((option: FilterOptionOption<ComboboxOption>, inputValue: string) => {
  if (!inputValue) {
    return true;
  }
  const regex = createRegex(inputValue);
  return (
    regex.test(option.label) ||
    regex.test(option.data.description ?? '') ||
    (option.data.tags ?? []).some((tag) => regex.test(tag))
  );
});

const AddNodePopover = () => {
  const dispatch = useAppDispatch();
  const buildInvocation = useBuildNode();
  const { t } = useTranslation();
  const selectRef = useRef<SelectInstance<ComboboxOption> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const templates = useStore($templates);
  const pendingConnection = useStore($pendingConnection);
  const isOpen = useStore($isAddNodePopoverOpen);
  const store = useAppStore();
  const isWorkflowsActive = useStore(INTERACTION_SCOPES.workflows.$isActive);

  const filteredTemplates = useMemo(() => {
    // If we have a connection in progress, we need to filter the node choices
    const templatesArray = map(templates);
    if (!pendingConnection) {
      return templatesArray;
    }

    return filter(templates, (template) => {
      const candidateFields = pendingConnection.handleType === 'source' ? template.inputs : template.outputs;
      return some(candidateFields, (field) => {
        const sourceType =
          pendingConnection.handleType === 'source' ? field.type : pendingConnection.fieldTemplate.type;
        const targetType =
          pendingConnection.handleType === 'target' ? field.type : pendingConnection.fieldTemplate.type;
        return validateConnectionTypes(sourceType, targetType);
      });
    });
  }, [templates, pendingConnection]);

  const options = useMemo(() => {
    const _options: ComboboxOption[] = map(filteredTemplates, (template) => {
      return {
        label: template.title,
        value: template.type,
        description: template.description,
        tags: template.tags,
      };
    });

    //We only want these nodes if we're not filtered
    if (!pendingConnection) {
      _options.push({
        label: t('nodes.currentImage'),
        value: 'current_image',
        description: t('nodes.currentImageDescription'),
        tags: ['progress'],
      });

      _options.push({
        label: t('nodes.notes'),
        value: 'notes',
        description: t('nodes.notesDescription'),
        tags: ['notes'],
      });
    }

    _options.sort((a, b) => a.label.localeCompare(b.label));

    return _options;
  }, [filteredTemplates, pendingConnection, t]);

  const addNode = useCallback(
    (nodeType: string): AnyNode | null => {
      const node = buildInvocation(nodeType);
      if (!node) {
        const errorMessage = t('nodes.unknownNode', {
          nodeType: nodeType,
        });
        toast({
          status: 'error',
          title: errorMessage,
        });
        return null;
      }

      // Find a cozy spot for the node
      const cursorPos = $cursorPos.get();
      const { nodes, edges } = store.getState().nodes.present;
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
        dispatch(nodesChanged(nodeChanges));
      }
      if (edgeChanges.length > 0) {
        dispatch(edgesChanged(edgeChanges));
      }
      return node;
    },
    [buildInvocation, store, dispatch, t]
  );

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      const node = addNode(v.value);

      // Auto-connect an edge if we just added a node and have a pending connection
      if (pendingConnection && isInvocationNode(node)) {
        const edgePendingUpdate = $edgePendingUpdate.get();
        const { handleType } = pendingConnection;

        const source = handleType === 'source' ? pendingConnection.nodeId : node.id;
        const sourceHandle = handleType === 'source' ? pendingConnection.handleId : null;
        const target = handleType === 'target' ? pendingConnection.nodeId : node.id;
        const targetHandle = handleType === 'target' ? pendingConnection.handleId : null;

        const { nodes, edges } = store.getState().nodes.present;
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
          dispatch(edgesChanged([{ type: 'add', item: newEdge }]));
        }
      }

      closeAddNodePopover();
    },
    [addNode, dispatch, pendingConnection, store, templates]
  );

  const handleHotkeyOpen: HotkeyCallback = useCallback((e) => {
    if (!$isAddNodePopoverOpen.get()) {
      e.preventDefault();
      openAddNodePopover();
      flushSync(() => {
        selectRef.current?.inputRef?.focus();
      });
    }
  }, []);

  useHotkeys(['shift+a', 'space'], handleHotkeyOpen, { enabled: isWorkflowsActive }, [isWorkflowsActive]);

  const noOptionsMessage = useCallback(() => t('nodes.noMatchingNodes'), [t]);

  return (
    <Popover
      isOpen={isOpen}
      onClose={closeAddNodePopover}
      placement="bottom"
      openDelay={0}
      closeDelay={0}
      closeOnBlur={true}
      returnFocusOnClose={true}
      initialFocusRef={inputRef}
      isLazy
    >
      <PopoverAnchor>
        <Flex position="absolute" top="15%" insetInlineStart="50%" pointerEvents="none" />
      </PopoverAnchor>
      <PopoverContent
        p={0}
        top={-1}
        shadow="dark-lg"
        borderColor="invokeBlue.400"
        borderWidth="2px"
        borderStyle="solid"
      >
        <PopoverBody w="32rem" p={0}>
          <Combobox
            menuIsOpen={isOpen}
            selectRef={selectRef}
            value={null}
            placeholder={t('nodes.nodeSearch')}
            options={options}
            noOptionsMessage={noOptionsMessage}
            filterOption={filterOption}
            onChange={onChange}
            onMenuClose={closeAddNodePopover}
            inputRef={inputRef}
            closeMenuOnSelect={false}
          />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(AddNodePopover);
