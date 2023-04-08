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
import {
  FIELDS,
  FIELD_NAMES,
  INVOCATIONS,
  INVOCATION_NAMES,
} from '../constants';
import { useAppDispatch } from 'app/storeHooks';
import { nodeAdded } from '../store/nodesSlice';
import { Flow } from './Flow';

const NodeEditor = () => {
  const dispatch = useAppDispatch();

  const addNode = useCallback(
    (nodeType: string) => {
      dispatch(
        nodeAdded({
          id: uuidv4(),
          type: nodeType,
          position: { x: 0, y: 0 },
          data: {},
        })
      );
    },
    [dispatch]
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
        {FIELD_NAMES.map((field) => (
          <Badge
            key={field}
            colorScheme={FIELDS[field].color}
            sx={{ userSelect: 'none' }}
          >
            {field}
          </Badge>
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
          {INVOCATION_NAMES.map((name) => {
            const invocation = INVOCATIONS[name];
            return (
              <Tooltip
                key={name}
                label={invocation.description}
                placement="end"
                hasArrow
              >
                <MenuItem onClick={() => addNode(invocation.title)}>
                  {invocation.title}
                </MenuItem>
              </Tooltip>
            );
          })}
        </MenuList>
      </Menu>
    </Box>
  );
};

export default NodeEditor;
