import { v4 as uuidv4 } from 'uuid';

import 'reactflow/dist/style.css';
import { useCallback } from 'react';
import {
  Tooltip,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  IconButton,
} from '@chakra-ui/react';
import { FaPlus } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { nodeAdded } from '../store/nodesSlice';
import { cloneDeep, map } from 'lodash';
import { RootState } from 'app/store';
import { useBuildInvocation } from '../hooks/useBuildInvocation';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/hooks/useToastWatcher';

export const AddNodeMenu = () => {
  const dispatch = useAppDispatch();

  const invocationTemplates = useAppSelector(
    (state: RootState) => state.nodes.invocationTemplates
  );

  const buildInvocation = useBuildInvocation();

  const addNode = useCallback(
    (nodeType: string) => {
      const invocation = buildInvocation(nodeType);

      if (!invocation) {
        const toast = makeToast({
          status: 'error',
          title: `Unknown Invocation type ${nodeType}`,
        });
        dispatch(addToast(toast));
        return;
      }

      dispatch(nodeAdded(invocation));
    },
    [dispatch, buildInvocation]
  );

  return (
    <Menu>
      <MenuButton as={IconButton} aria-label="Add Node" icon={<FaPlus />} />
      <MenuList overflowY="scroll" height={400}>
        {map(invocationTemplates, ({ title, description, type }, key) => {
          return (
            <Tooltip key={key} label={description} placement="end" hasArrow>
              <MenuItem onClick={() => addNode(type)}>{title}</MenuItem>
            </Tooltip>
          );
        })}
      </MenuList>
    </Menu>
  );
};
