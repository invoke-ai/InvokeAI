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
import { map } from 'lodash';
import { RootState } from 'app/store';

export const AddNodeMenu = () => {
  const dispatch = useAppDispatch();

  const invocations = useAppSelector(
    (state: RootState) => state.nodes.invocations
  );

  const addNode = useCallback(
    (nodeType: string) => {
      dispatch(nodeAdded({ id: uuidv4(), invocation: invocations[nodeType] }));
    },
    [dispatch, invocations]
  );

  return (
    <Menu>
      <MenuButton as={IconButton} aria-label="Add Node" icon={<FaPlus />} />
      <MenuList>
        {map(invocations, ({ title, description, type }, key) => {
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
