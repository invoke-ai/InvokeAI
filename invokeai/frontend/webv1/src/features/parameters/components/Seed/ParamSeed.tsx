import type { FlexProps } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { ParamSeedNumberInput } from 'features/parameters/components/Seed/ParamSeedNumberInput';
import { ParamSeedRandomize } from 'features/parameters/components/Seed/ParamSeedRandomize';
import { ParamSeedShuffle } from 'features/parameters/components/Seed/ParamSeedShuffle';
import { memo } from 'react';

export const ParamSeed = memo((props: FlexProps) => {
  return (
    <Flex gap={4} alignItems="center" {...props}>
      <ParamSeedNumberInput />
      <ParamSeedRandomize />
      <ParamSeedShuffle />
    </Flex>
  );
});

ParamSeed.displayName = 'ParamSeed';
