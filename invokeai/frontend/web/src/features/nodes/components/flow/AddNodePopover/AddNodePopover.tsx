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
import { validateSourceAndTargetTypes } from 'features/nodes/store/util/validateSourceAndTargetTypes';
import { filter, map, some } from 'lodash-es';
import { memo, useCallback, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { HotkeyCallback } from 'react-hotkeys-hook/dist/types';
import { useTranslation } from 'react-i18next';
import 'reactflow/dist/style.css';
import { AnyInvocationType } from 'services/events/types';
import { AddNodePopoverSelectItem } from './AddNodePopoverSelectItem';

type NodeTemplate = {
  label: string;
  value: string;
  description: string;
  tags: string[];
};

const selectFilter = (value: string, item: NodeTemplate) => {
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

const AddNodePopover = () => {
  const dispatch = useAppDispatch();
  const buildInvocation = useBuildNodeData();
  const toaster = useAppToaster();
  const { t } = useTranslation();

  const fieldFilter = useAppSelector(
    (state) => state.nodes.currentConnectionFieldType
  );
  const handleFilter = useAppSelector(
    (state) => state.nodes.connectionStartParams?.handleType
  );

  const selector = createSelector(
    [stateSelector],
    ({ nodes }) => {
      // If we have a connection in progress, we need to filter the node choices
      const filteredNodeTemplates = fieldFilter
        ? filter(nodes.nodeTemplates, (template) => {
            const handles =
              handleFilter == 'source' ? template.inputs : template.outputs;

            return some(handles, (handle) => {
              const sourceType =
                handleFilter == 'source'
                  ? fieldFilter
                  : handle.originalType ?? handle.type;
              const targetType =
                handleFilter == 'target'
                  ? fieldFilter
                  : handle.originalType ?? handle.type;

              return validateSourceAndTargetTypes(sourceType, targetType);
            });
          })
        : map(nodes.nodeTemplates);

      const data: NodeTemplate[] = map(filteredNodeTemplates, (template) => {
        return {
          label: template.title,
          value: template.type,
          description: template.description,
          tags: template.tags,
        };
      });

      //We only want these nodes if we're not filtered
      if (fieldFilter === null) {
        data.push({
          label: t('nodes.currentImage'),
          value: 'current_image',
          description: t('nodes.currentImageDescription'),
          tags: ['progress'],
        });

        data.push({
          label: t('nodes.notes'),
          value: 'notes',
          description: t('nodes.notesDescription'),
          tags: ['notes'],
        });
      }

      data.sort((a, b) => a.label.localeCompare(b.label));

      return { data };
    },
    defaultSelectorOptions
  );

  const { data } = useAppSelector(selector);
  const isOpen = useAppSelector((state) => state.nodes.isAddNodePopoverOpen);
  const inputRef = useRef<HTMLInputElement>(null);

  const addNode = useCallback(
    (nodeType: AnyInvocationType) => {
      const invocation = buildInvocation(nodeType);
      if (!invocation) {
        const errorMessage = t('nodes.unknownNode', {
          nodeType: nodeType,
        });
        toaster({
          status: 'error',
          title: errorMessage,
        });
        return;
      }

      dispatch(nodeAdded(invocation));
    },
    [dispatch, buildInvocation, toaster, t]
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
            placeholder={t('nodes.nodeSearch')}
            value={null}
            data={data}
            maxDropdownHeight={400}
            nothingFound={t('nodes.noMatchingNodes')}
            itemComponent={AddNodePopoverSelectItem}
            filter={selectFilter}
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
