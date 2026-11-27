import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectKrea2RebalanceEnabled, selectKrea2SeedVarianceEnabled } from 'features/controlLayers/store/paramsSlice';
import { memo } from 'react';

import ParamKrea2RebalanceEnabled from './ParamKrea2RebalanceEnabled';
import ParamKrea2RebalanceMultiplier from './ParamKrea2RebalanceMultiplier';
import ParamKrea2RebalanceWeights from './ParamKrea2RebalanceWeights';
import ParamKrea2SeedVarianceEnabled from './ParamKrea2SeedVarianceEnabled';
import ParamKrea2SeedVarianceRandomizePercent from './ParamKrea2SeedVarianceRandomizePercent';
import ParamKrea2SeedVarianceStrength from './ParamKrea2SeedVarianceStrength';

const ParamKrea2EnhancersSettings = () => {
  const rebalanceEnabled = useAppSelector(selectKrea2RebalanceEnabled);
  const seedVarianceEnabled = useAppSelector(selectKrea2SeedVarianceEnabled);

  return (
    <Flex gap={4} flexDir="column" w="full">
      <ParamKrea2RebalanceEnabled />
      {rebalanceEnabled && (
        <>
          <ParamKrea2RebalanceMultiplier />
          <ParamKrea2RebalanceWeights />
        </>
      )}
      <Divider />
      <ParamKrea2SeedVarianceEnabled />
      {seedVarianceEnabled && (
        <>
          <ParamKrea2SeedVarianceStrength />
          <ParamKrea2SeedVarianceRandomizePercent />
        </>
      )}
    </Flex>
  );
};

export default memo(ParamKrea2EnhancersSettings);
