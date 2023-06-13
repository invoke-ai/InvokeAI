import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ParamSeed from './ParamSeed';
import ParamSeedShuffle from './ParamSeedShuffle';
import ParamSeedRandomize from './ParamSeedRandomize';

const ParamSeedFull = () => {
  return (
    <Flex sx={{ gap: 4, alignItems: 'center' }}>
      <ParamSeed />
      <ParamSeedShuffle />
      <ParamSeedRandomize />
    </Flex>
  );
};

export default memo(ParamSeedFull);
