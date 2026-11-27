import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectZImageSeedVarianceEnabled } from 'features/controlLayers/store/paramsSlice';
import { memo } from 'react';

import ParamZImageSeedVarianceEnabled from './ParamZImageSeedVarianceEnabled';
import ParamZImageSeedVarianceRandomizePercent from './ParamZImageSeedVarianceRandomizePercent';
import ParamZImageSeedVarianceStrength from './ParamZImageSeedVarianceStrength';

const ParamZImageSeedVarianceSettings = () => {
  const enabled = useAppSelector(selectZImageSeedVarianceEnabled);

  return (
    <Flex gap={4} flexDir="column">
      <ParamZImageSeedVarianceEnabled />
      {enabled && (
        <>
          <ParamZImageSeedVarianceStrength />
          <ParamZImageSeedVarianceRandomizePercent />
        </>
      )}
    </Flex>
  );
};

export default memo(ParamZImageSeedVarianceSettings);
