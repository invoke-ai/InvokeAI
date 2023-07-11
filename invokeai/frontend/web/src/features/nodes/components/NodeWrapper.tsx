import { Box, useToken } from '@chakra-ui/react';
import { NODE_MIN_WIDTH } from 'app/constants';

import { PropsWithChildren } from 'react';

type NodeWrapperProps = PropsWithChildren & {
  selected: boolean;
};

const NodeWrapper = (props: NodeWrapperProps) => {
  const [nodeSelectedOutline, nodeShadow] = useToken('shadows', [
    'nodeSelectedOutline',
    'dark-lg',
  ]);

  return (
    <Box
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
