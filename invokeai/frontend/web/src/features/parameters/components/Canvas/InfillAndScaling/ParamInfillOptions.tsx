import { useAppSelector } from 'app/store/storeHooks';

import ParamInfillPatchmatchDownscaleSize from './ParamInfillPatchmatchDownscaleSize';
import ParamInfillTilesize from './ParamInfillTilesize';

export default function ParamInfillOptions() {
  const infillMethod = useAppSelector((state) => state.generation.infillMethod);
  if (infillMethod === 'tile') {
    return <ParamInfillTilesize />;
  }

  if (infillMethod === 'patchmatch') {
    return <ParamInfillPatchmatchDownscaleSize />;
  }

  return null;
}
