import 'reactflow/dist/style.css';

import { Flex } from '@chakra-ui/react';
import { useAppToaster } from 'app/components/Toaster';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { SelectInstance } from 'chakra-react-select';
import {
  InvPopover,
  InvPopoverAnchor,
  InvPopoverBody,
  InvPopoverContent,
} from 'common/components/InvPopover/wrapper';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { useBuildNode } from 'features/nodes/hooks/useBuildNode';
import {
  addNodePopoverClosed,
  addNodePopoverOpened,
  nodeAdded,
} from 'features/nodes/store/nodesSlice';
import { validateSourceAndTargetTypes } from 'features/nodes/store/util/validateSourceAndTargetTypes';
import { filter, map, memoize, some } from 'lodash-es';
import type { KeyboardEventHandler } from 'react';
import { memo, useCallback, useRef } from 'react';
import { flushSync } from 'react-dom';
import { useHotkeys } from 'react-hotkeys-hook';
import type { HotkeyCallback } from 'react-hotkeys-hook/dist/types';
import { useTranslation } from 'react-i18next';
import type { FilterOptionOption } from 'react-select/dist/declarations/src/filters';

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

const filterOption = memoize(
  (option: FilterOptionOption<InvSelectOption>, inputValue: string) => {
    if (!inputValue) {
      return true;
    }
    const regex = createRegex(inputValue);
    return (
      regex.test(option.label) ||
      regex.test(option.data.description ?? '') ||
      (option.data.tags ?? []).some((tag) => regex.test(tag))
    );
  }
);

const AddNodePopover = () => {
  const dispatch = useAppDispatch();
  const buildInvocation = useBuildNode();
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const selectRef = useRef<SelectInstance<InvSelectOption> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const fieldFilter = useAppSelector(
    (state) => state.nodes.connectionStartFieldType
  );
  const handleFilter = useAppSelector(
    (state) => state.nodes.connectionStartParams?.handleType
  );

  const selector = createMemoizedSelector([stateSelector], ({ nodes }) => {
    // If we have a connection in progress, we need to filter the node choices
    const filteredNodeTemplates = fieldFilter
      ? filter(nodes.nodeTemplates, (template) => {
          const handles =
            handleFilter == 'source' ? template.inputs : template.outputs;

          return some(handles, (handle) => {
            const sourceType =
              handleFilter == 'source' ? fieldFilter : handle.type;
            const targetType =
              handleFilter == 'target' ? fieldFilter : handle.type;

            return validateSourceAndTargetTypes(sourceType, targetType);
          });
        })
      : map(nodes.nodeTemplates);

    const options: InvSelectOption[] = map(
      filteredNodeTemplates,
      (template) => {
        return {
          label: template.title,
          value: template.type,
          description: template.description,
          tags: template.tags,
        };
      }
    );

    //We only want these nodes if we're not filtered
    if (fieldFilter === null) {
      options.push({
        label: t('nodes.currentImage'),
        value: 'current_image',
        description: t('nodes.currentImageDescription'),
        tags: ['progress'],
      });

      options.push({
        label: t('nodes.notes'),
        value: 'notes',
        description: t('nodes.notesDescription'),
        tags: ['notes'],
      });
    }

    options.sort((a, b) => a.label.localeCompare(b.label));

    return { options };
  });

  const { options } = useAppSelector(selector);
  const isOpen = useAppSelector((state) => state.nodes.isAddNodePopoverOpen);

  const addNode = useCallback(
    (nodeType: string) => {
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

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      addNode(v.value);
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
      flushSync(() => {
        selectRef.current?.inputRef?.focus();
      });
    },
    [onOpen]
  );

  const handleHotkeyClose: HotkeyCallback = useCallback(() => {
    onClose();
  }, [onClose]);

  useHotkeys(['shift+a', 'space'], handleHotkeyOpen);
  useHotkeys(['escape'], handleHotkeyClose);
  const onKeyDown: KeyboardEventHandler = useCallback(
    (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    },
    [onClose]
  );

  const noOptionsMessage = useCallback(() => t('nodes.noMatchingNodes'), [t]);

  return (
    <InvPopover
      isOpen={isOpen}
      onClose={onClose}
      placement="bottom"
      openDelay={0}
      closeDelay={0}
      closeOnBlur={true}
      returnFocusOnClose={true}
      initialFocusRef={inputRef}
    >
      <InvPopoverAnchor>
        <Flex
          position="absolute"
          top="15%"
          insetInlineStart="50%"
          pointerEvents="none"
        />
      </InvPopoverAnchor>
      <InvPopoverContent
        p={0}
        top={-1}
        shadow="dark-lg"
        borderColor="blue.400"
        borderWidth="2px"
        borderStyle="solid"
      >
        <InvPopoverBody w="32rem" p={0}>
          <InvSelect
            menuIsOpen={isOpen}
            selectRef={selectRef}
            value={null}
            placeholder={t('nodes.nodeSearch')}
            options={options}
            noOptionsMessage={noOptionsMessage}
            filterOption={filterOption}
            onChange={onChange}
            onMenuClose={onClose}
            onKeyDown={onKeyDown}
            inputRef={inputRef}
          />
        </InvPopoverBody>
      </InvPopoverContent>
    </InvPopover>
  );
};

export default memo(AddNodePopover);
