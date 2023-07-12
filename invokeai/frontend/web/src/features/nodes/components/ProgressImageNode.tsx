import { Flex, Image } from '@chakra-ui/react';
import { NodeProps } from 'reactflow';
import { InvocationValue } from '../types/types';

import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo } from 'react';
import IAINodeHeader from './IAINode/IAINodeHeader';
import NodeWrapper from './NodeWrapper';

const ProgressImageNode = (props: NodeProps<InvocationValue>) => {
  const progressImage = useAppSelector((state) => state.system.progressImage);
  const { selected } = props;

  return (
    <NodeWrapper selected={selected}>
      <IAINodeHeader
        title="Progress Image"
        description="Displays the progress image in the Node Editor"
      />

      <Flex
        sx={{
          flexDirection: 'column',
          borderBottomRadius: 'md',
          p: 2,
          bg: 'base.200',
          _dark: { bg: 'base.800' },
          width: '384px',
          height: '384px',
        }}
      >
        {progressImage ? (
          <Image
            src={progressImage.dataURL}
            sx={{
              w: 'full',
              h: 'full',
              objectFit: 'contain',
            }}
          />
        ) : (
          <Flex
            sx={{
              w: 'full',
              h: 'full',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <IAINoContentFallback />
          </Flex>
        )}
      </Flex>
    </NodeWrapper>
  );
};

export default memo(ProgressImageNode);
