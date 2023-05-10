import { v4 as uuidv4 } from 'uuid';

import 'reactflow/dist/style.css';
import { memo, useCallback } from 'react';
import {
  Tooltip,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  IconButton,
} from '@chakra-ui/react';
import { FaEllipsisV, FaPlus } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { nodeAdded } from '../store/nodesSlice';
import { cloneDeep, map } from 'lodash-es';
import { RootState } from 'app/store/store';
import { useBuildInvocation } from '../hooks/useBuildInvocation';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/hooks/useToastWatcher';
import { AnyInvocationType } from 'services/events/types';
import IAIIconButton from 'common/components/IAIIconButton';

const AddNodeMenu = () => {
  const dispatch = useAppDispatch();

  const invocationTemplates = useAppSelector(
    (state: RootState) => state.nodes.invocationTemplates
  );

  const buildInvocation = useBuildInvocation();

  const addNode = useCallback(
    (nodeType: AnyInvocationType) => {
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
    <Menu isLazy>
      <MenuButton
        as={IAIIconButton}
        aria-label="Add Node"
        icon={<FaEllipsisV />}
      />
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

export default memo(AddNodeMenu);
