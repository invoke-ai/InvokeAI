import 'reactflow/dist/style.css';
import { memo, useCallback } from 'react';
import {
  Tooltip,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from '@chakra-ui/react';
import { FaEllipsisV } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { nodeAdded } from '../store/nodesSlice';
import { map } from 'lodash-es';
import { RootState } from 'app/store/store';
import { useBuildInvocation } from '../hooks/useBuildInvocation';
import { AnyInvocationType } from 'services/events/types';
import IAIIconButton from 'common/components/IAIIconButton';
import { useAppToaster } from 'app/components/Toaster';

const AddNodeMenu = () => {
  const dispatch = useAppDispatch();

  const invocationTemplates = useAppSelector(
    (state: RootState) => state.nodes.invocationTemplates
  );

  const buildInvocation = useBuildInvocation();

  const toaster = useAppToaster();

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
