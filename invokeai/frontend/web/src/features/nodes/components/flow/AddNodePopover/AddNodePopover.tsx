import {
  Flex,
  Popover,
  PopoverAnchor,
  PopoverBody,
  PopoverContent,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppToaster } from 'app/components/Toaster';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { useBuildNodeData } from 'features/nodes/hooks/useBuildNodeData';
import {
  addNodePopoverClosed,
  addNodePopoverOpened,
  nodeAdded,
} from 'features/nodes/store/nodesSlice';
import { map } from 'lodash-es';
import { memo, useCallback, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { HotkeyCallback } from 'react-hotkeys-hook/dist/types';
import 'reactflow/dist/style.css';
import { AnyInvocationType } from 'services/events/types';
import { AddNodePopoverSelectItem } from './AddNodePopoverSelectItem';

type NodeTemplate = {
  label: string;
  value: string;
  description: string;
  tags: string[];
};

const filter = (value: string, item: NodeTemplate) => {
  const regex = new RegExp(
    value
      .trim()
      .replace(/[-[\]{}()*+!<=:?./\\^$|#,]/g, '')
      .split(' ')
      .join('.*'),
    'gi'
  );
  return (
    regex.test(item.label) ||
    regex.test(item.description) ||
    item.tags.some((tag) => regex.test(tag))
  );
};

const selector = createSelector(
  [stateSelector],
  ({ nodes }) => {
    const data: NodeTemplate[] = map(nodes.nodeTemplates, (template) => {
      return {
        label: template.title,
        value: template.type,
        description: template.description,
        tags: template.tags,
      };
    });

    data.push({
      label: 'Progress Image',
      value: 'current_image',
      description: 'Displays the current image in the Node Editor',
      tags: ['progress'],
    });

    data.push({
      label: 'Notes',
      value: 'notes',
      description: 'Add notes about your workflow',
      tags: ['notes'],
    });

    data.sort((a, b) => a.label.localeCompare(b.label));

    return { data };
  },
  defaultSelectorOptions
);

const AddNodePopover = () => {
  const dispatch = useAppDispatch();
  const buildInvocation = useBuildNodeData();
  const toaster = useAppToaster();
  const { data } = useAppSelector(selector);
  const isOpen = useAppSelector((state) => state.nodes.isAddNodePopoverOpen);
  const inputRef = useRef<HTMLInputElement>(null);

  const addNode = useCallback(
    (nodeType: AnyInvocationType) => {
      const invocation = buildInvocation(nodeType);

      if (!invocation) {
        toaster({
          status: 'error',
          title: `Unknown Invocation type ${nodeType}`,
        });
        return;
      }

      dispatch(nodeAdded(invocation));
    },
    [dispatch, buildInvocation, toaster]
  );

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      addNode(v as AnyInvocationType);
    },
    [addNode]
  );

  const onClose = useCallback(() => {
    dispatch(addNodePopoverClosed());
  }, [dispatch]);

  const onOpen = useCallback(() => {
    dispatch(addNodePopoverOpened());
  }, [dispatch]);

  const handleHotkeyOpen: HotkeyCallback = useCallback(
    (e) => {
      e.preventDefault();
      onOpen();
      setTimeout(() => {
        inputRef.current?.focus();
      }, 0);
    },
    [onOpen]
  );

  const handleHotkeyClose: HotkeyCallback = useCallback(() => {
    onClose();
  }, [onClose]);

  useHotkeys(['shift+a', 'space'], handleHotkeyOpen);
  useHotkeys(['escape'], handleHotkeyClose);

  return (
    <Popover
      initialFocusRef={inputRef}
      isOpen={isOpen}
      onClose={onClose}
      placement="bottom"
      openDelay={0}
      closeDelay={0}
      closeOnBlur={true}
      returnFocusOnClose={true}
    >
      <PopoverAnchor>
        <Flex
          sx={{
            position: 'absolute',
            top: '15%',
            insetInlineStart: '50%',
            pointerEvents: 'none',
          }}
        />
      </PopoverAnchor>
      <PopoverContent
        sx={{
          p: 0,
          top: -1,
          shadow: 'dark-lg',
          borderColor: 'accent.300',
          borderWidth: '2px',
          borderStyle: 'solid',
          _dark: { borderColor: 'accent.400' },
        }}
      >
        <PopoverBody sx={{ p: 0 }}>
          <IAIMantineSearchableSelect
            inputRef={inputRef}
            selectOnBlur={false}
            placeholder="Search for nodes"
            value={null}
            data={data}
            maxDropdownHeight={400}
            nothingFound="No matching nodes"
            itemComponent={AddNodePopoverSelectItem}
            filter={filter}
            onChange={handleChange}
            hoverOnSearchChange={true}
            onDropdownClose={onClose}
            sx={{
              width: '32rem',
              input: {
                padding: '0.5rem',
              },
            }}
          />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(AddNodePopover);
