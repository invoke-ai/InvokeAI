import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectImg2imgStrength, setImg2imgStrength } from 'features/controlLayers/store/paramsSlice';
import ImageToImageStrength from 'features/parameters/components/ImageToImage/ImageToImageStrength';
import { memo, useCallback } from 'react';

const ParamImageToImageStrength = () => {
  const img2imgStrength = useAppSelector(selectImg2imgStrength);
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
