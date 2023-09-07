import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import ParamInfillPatchmatchDownscaleSize from './ParamInfillPatchmatchDownscaleSize';
import ParamInfillTilesize from './ParamInfillTilesize';

const selector = createSelector(
  [generationSelector],
  (parameters) => {
    const { infillMethod } = parameters;

    return {
      infillMethod,
    };
  },
  defaultSelectorOptions
);

export default function ParamInfillOptions() {
  const { infillMethod } = useAppSelector(selector);
  return (
    <Flex>
      {infillMethod === 'tile' && <ParamInfillTilesize />}
      {infillMethod === 'patchmatch' && <ParamInfillPatchmatchDownscaleSize />}
    </Flex>
  );
}
