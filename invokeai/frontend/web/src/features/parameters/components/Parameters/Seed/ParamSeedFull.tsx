import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ParamSeed from './ParamSeed';
import ParamSeedShuffle from './ParamSeedShuffle';
import ParamSeedRandomize from './ParamSeedRandomize';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';

const ParamSeedFull = () => {
  return (
    <IAIInformationalPopover feature="paramSeed">
      <Flex sx={{ gap: 3, alignItems: 'flex-end' }}>
        <ParamSeed />
        <ParamSeedShuffle />
        <ParamSeedRandomize />
      </Flex>
    </IAIInformationalPopover>
  );
};

export default memo(ParamSeedFull);
