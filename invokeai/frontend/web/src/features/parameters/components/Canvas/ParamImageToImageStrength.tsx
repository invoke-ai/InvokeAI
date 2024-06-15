import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setImg2imgStrength } from 'features/controlLayers/store/canvasV2Slice';
import ImageToImageStrength from 'features/parameters/components/ImageToImage/ImageToImageStrength';
import { memo, useCallback } from 'react';

const ParamImageToImageStrength = () => {
  const img2imgStrength = useAppSelector((s) => s.canvasV2.params.img2imgStrength);
  const dispatch = useAppDispatch();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setImg2imgStrength(v));
    },
    [dispatch]
  );

  return <ImageToImageStrength value={img2imgStrength} onChange={onChange} />;
};

export default memo(ParamImageToImageStrength);
