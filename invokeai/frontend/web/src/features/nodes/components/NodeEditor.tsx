import { v4 as uuidv4 } from 'uuid';

import 'reactflow/dist/style.css';
import { useCallback } from 'react';
import {
  Box,
  Tooltip,
  Badge,
  HStack,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  IconButton,
} from '@chakra-ui/react';
import { FaPlus } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { nodeAdded } from '../store/nodesSlice';
import { Flow } from './Flow';
import { FIELDS } from '../types';
import { map } from 'lodash';
import { RootState } from 'app/store';

const NodeEditor = () => {
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
    <Box
      sx={{
        position: 'relative',
        width: 'full',
        height: 'full',
        borderRadius: 'md',
        bg: 'base.850',
      }}
    >
      <Flow />
      <HStack sx={{ position: 'absolute', top: 2, right: 2 }}>
        {map(FIELDS, ({ title, description, color }, key) => (
          <Tooltip key={key} label={description}>
            <Badge colorScheme={color} sx={{ userSelect: 'none' }}>
              {title}
            </Badge>
          </Tooltip>
        ))}
      </HStack>
      <Menu>
        <MenuButton
          as={IconButton}
          aria-label="Options"
          icon={<FaPlus />}
          sx={{ position: 'absolute', top: 2, left: 2 }}
        />
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
    </Box>
  );
};

export default NodeEditor;
