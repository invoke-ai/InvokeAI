import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ParamInfillPatchmatchDownscaleSize from './ParamInfillPatchmatchDownscaleSize';
import ParamInfillTilesize from './ParamInfillTilesize';

const selector = createMemoizedSelector([stateSelector], ({ generation }) => {
  const { infillMethod } = generation;

  return {
    infillMethod,
  };
});

export default function ParamInfillOptions() {
  const { infillMethod } = useAppSelector(selector);
  return (
    <Flex>
      {infillMethod === 'tile' && <ParamInfillTilesize />}
      {infillMethod === 'patchmatch' && <ParamInfillPatchmatchDownscaleSize />}
    </Flex>
  );
}
