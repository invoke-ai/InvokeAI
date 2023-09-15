import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ParamSeed from './ParamSeed';
import ParamSeedShuffle from './ParamSeedShuffle';
import ParamSeedRandomize from './ParamSeedRandomize';
import { ParamSeedPopover } from 'features/informationalPopovers/components/paramSeed';

const ParamSeedFull = () => {
  return (
    <ParamSeedPopover>
      <Flex sx={{ gap: 3, alignItems: 'flex-end' }}>
        <ParamSeed />
        <ParamSeedShuffle />
        <ParamSeedRandomize />
      </Flex>
    </ParamSeedPopover>
  );
};

export default memo(ParamSeedFull);
