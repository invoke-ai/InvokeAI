import { Box, useToken } from '@chakra-ui/react';
import { NODE_MIN_WIDTH } from 'app/constants';

import { useAppSelector } from 'app/store/storeHooks';
import { PropsWithChildren } from 'react';
import { DRAG_HANDLE_CLASSNAME } from '../hooks/useBuildInvocation';

type NodeWrapperProps = PropsWithChildren & {
  selected: boolean;
};

const NodeWrapper = (props: NodeWrapperProps) => {
  const [nodeSelectedOutline, nodeShadow] = useToken('shadows', [
    'nodeSelectedOutline',
    'dark-lg',
  ]);

  const shift = useAppSelector((state) => state.hotkeys.shift);

  return (
    <Box
      className={shift ? DRAG_HANDLE_CLASSNAME : 'nopan'}
      sx={{
        position: 'relative',
        borderRadius: 'md',
        minWidth: NODE_MIN_WIDTH,
        shadow: props.selected
          ? `${nodeSelectedOutline}, ${nodeShadow}`
          : `${nodeShadow}`,
      }}
    >
      {props.children}
    </Box>
  );
};

export default NodeWrapper;
