import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';

import ParamInfillPatchmatchDownscaleSize from './ParamInfillPatchmatchDownscaleSize';
import ParamInfillTilesize from './ParamInfillTilesize';

const ParamInfillOptions = () => {
  const infillMethod = useAppSelector((s) => s.generation.infillMethod);
  if (infillMethod === 'tile') {
    return <ParamInfillTilesize />;
  }

  if (infillMethod === 'patchmatch') {
    return <ParamInfillPatchmatchDownscaleSize />;
  }

  return null;
};

export default memo(ParamInfillOptions);
