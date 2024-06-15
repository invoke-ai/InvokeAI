import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';

import ParamInfillColorOptions from './ParamInfillColorOptions';
import ParamInfillPatchmatchDownscaleSize from './ParamInfillPatchmatchDownscaleSize';
import ParamInfillTilesize from './ParamInfillTilesize';

const ParamInfillOptions = () => {
  const infillMethod = useAppSelector((s) => s.canvasV2.compositing.infillMethod);
  if (infillMethod === 'tile') {
    return <ParamInfillTilesize />;
  }

  if (infillMethod === 'patchmatch') {
    return <ParamInfillPatchmatchDownscaleSize />;
  }

  if (infillMethod === 'color') {
    return <ParamInfillColorOptions />;
  }

  return null;
};

export default memo(ParamInfillOptions);
