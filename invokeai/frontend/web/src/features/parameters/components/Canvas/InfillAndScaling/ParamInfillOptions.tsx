import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';

import ParamInfillColorOptions from './ParamInfillColorOptions';
import ParamInfillMosaicOptions from './ParamInfillMosaicOptions';
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

  if (infillMethod === 'mosaic') {
    return <ParamInfillMosaicOptions />;
  }

  if (infillMethod === 'color') {
    return <ParamInfillColorOptions />;
  }

  return null;
};

export default memo(ParamInfillOptions);
