import { Box, Flex } from '@chakra-ui/layout';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInput from 'common/components/IAIInput';
import { map } from 'lodash-es';
import {
  ChangeEvent,
  FocusEvent,
  KeyboardEvent,
  memo,
  ReactNode,
  useCallback,
  useRef,
  useState,
} from 'react';
import { Tooltip } from '@chakra-ui/tooltip';
import { AnyInvocationType } from 'services/events/types';
import { useBuildInvocation } from 'features/nodes/hooks/useBuildInvocation';
import { addToast } from 'features/system/store/systemSlice';
import { nodeAdded } from '../../store/nodesSlice';
import Fuse from 'fuse.js';
import { InvocationTemplate } from 'features/nodes/types/types';
import { useAppToaster } from 'app/components/Toaster';

interface NodeListItemProps {
  title: string;
  description: string;
  type: AnyInvocationType;
  isSelected: boolean;
  addNode: (nodeType: AnyInvocationType) => void;
}

const NodeListItem = (props: NodeListItemProps) => {
  const { title, description, type, isSelected, addNode } = props;
  return (
    <Tooltip label={description} placement="end" hasArrow>
      <Box
        px={4}
        onClick={() => addNode(type)}
        background={isSelected ? 'base.600' : 'none'}
        _hover={{
          background: 'base.600',
          cursor: 'pointer',
        }}
      >
        {title}
      </Box>
    </Tooltip>
  );
};

NodeListItem.displayName = 'NodeListItem';

const NodeSearch = () => {
  const invocationTemplates = useAppSelector(
    (state: RootState) => state.nodes.invocationTemplates
  );

  const nodes = map(invocationTemplates);
  const [filteredNodes, setFilteredNodes] = useState<
    Fuse.FuseResult<InvocationTemplate>[]
  >([]);

  const buildInvocation = useBuildInvocation();
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();

  const [searchText, setSearchText] = useState<string>('');
  const [showNodeList, setShowNodeList] = useState<boolean>(false);
  const [focusedIndex, setFocusedIndex] = useState<number>(-1);
  const nodeSearchRef = useRef<HTMLDivElement>(null);

  const fuseOptions = {
    findAllMatches: true,
    threshold: 0,
    ignoreLocation: true,
    keys: ['title', 'type', 'tags'],
  };

  const fuse = new Fuse(nodes, fuseOptions);

  const findNode = (e: ChangeEvent<HTMLInputElement>) => {
    setSearchText(e.target.value);
    setFilteredNodes(fuse.search(e.target.value));
    setShowNodeList(true);
  };

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

  const renderNodeList = () => {
    const nodeListToRender: ReactNode[] = [];

    if (searchText.length > 0) {
      filteredNodes.forEach(({ item }, index) => {
        const { title, description, type } = item;
        if (title.toLowerCase().includes(searchText)) {
          nodeListToRender.push(
            <NodeListItem
              key={index}
              title={title}
              description={description}
              type={type}
              isSelected={focusedIndex === index}
              addNode={addNode}
            />
          );
        }
      });
    } else {
      nodes.forEach(({ title, description, type }, index) => {
        nodeListToRender.push(
          <NodeListItem
            key={index}
            title={title}
            description={description}
            type={type}
            isSelected={focusedIndex === index}
            addNode={addNode}
          />
        );
      });
    }

    return (
      <Flex flexDirection="column" background="base.900" borderRadius={6}>
        {nodeListToRender}
      </Flex>
    );
  };

  const searchKeyHandler = (e: KeyboardEvent<HTMLDivElement>) => {
    const { key } = e;
    let nextIndex = 0;

    if (key === 'ArrowDown') {
      setShowNodeList(true);
      if (searchText.length > 0) {
        nextIndex = (focusedIndex + 1) % filteredNodes.length;
      } else {
        nextIndex = (focusedIndex + 1) % nodes.length;
      }
    }

    if (key === 'ArrowUp') {
      setShowNodeList(true);
      if (searchText.length > 0) {
        nextIndex =
          (focusedIndex + filteredNodes.length - 1) % filteredNodes.length;
      } else {
        nextIndex = (focusedIndex + nodes.length - 1) % nodes.length;
      }
    }

    // # TODO Handle Blur
    // if (key === 'Escape') {
    // }

    if (key === 'Enter') {
      let selectedNodeType: AnyInvocationType;

      if (searchText.length > 0) {
        selectedNodeType = filteredNodes[focusedIndex].item.type;
      } else {
        selectedNodeType = nodes[focusedIndex].type;
      }

      addNode(selectedNodeType);
      setShowNodeList(false);
    }

    setFocusedIndex(nextIndex);
  };

  const searchInputBlurHandler = (e: FocusEvent<HTMLDivElement>) => {
    if (!e.currentTarget.contains(e.relatedTarget)) setShowNodeList(false);
  };

  return (
    <Flex
      flexDirection="column"
      tabIndex={1}
      onKeyDown={searchKeyHandler}
      onFocus={() => setShowNodeList(true)}
      onBlur={searchInputBlurHandler}
      ref={nodeSearchRef}
    >
      <IAIInput value={searchText} onChange={findNode} />
      {showNodeList && renderNodeList()}
    </Flex>
  );
};

export default memo(NodeSearch);
