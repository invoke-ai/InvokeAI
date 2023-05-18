import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldRunESRGAN } from 'features/parameters/store/postprocessingSlice';
import { ChangeEvent } from 'react';

export default function UpscaleToggle() {
  const isESRGANAvailable = useAppSelector(
    (state: RootState) => state.system.isESRGANAvailable
  );

  const shouldRunESRGAN = useAppSelector(
    (state: RootState) => state.postprocessing.shouldRunESRGAN
  );

  const dispatch = useAppDispatch();
  const handleChangeShouldRunESRGAN = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRunESRGAN(e.target.checked));
  return (
    <IAISwitch
      isDisabled={!isESRGANAvailable}
      isChecked={shouldRunESRGAN}
      onChange={handleChangeShouldRunESRGAN}
    />
  );
}
