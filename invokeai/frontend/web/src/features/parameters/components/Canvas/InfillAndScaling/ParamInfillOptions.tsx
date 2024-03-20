import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';

import ParamInfillPatchmatchDownscaleSize from './ParamInfillPatchmatchDownscaleSize';
import ParamInfillTilesize from './ParamInfillTilesize';
import ParamMosaicInfillOptions from './ParamMosaicInfillOptions';

const ParamInfillOptions = () => {
  const infillMethod = useAppSelector((s) => s.generation.infillMethod);
  if (infillMethod === 'tile') {
    return <ParamInfillTilesize />;
  }

  if (infillMethod === 'patchmatch') {
    return <ParamInfillPatchmatchDownscaleSize />;
  }

  if (infillMethod === 'mosaic') {
    return <ParamMosaicInfillOptions />;
  }

  return null;
};

export default memo(ParamInfillOptions);
