import { ChangeEvent } from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldRunESRGAN } from 'features/options/store/optionsSlice';

export default function UpscaleToggle() {
  const isESRGANAvailable = useAppSelector(
    (state: RootState) => state.system.isESRGANAvailable
  );

  const shouldRunESRGAN = useAppSelector(
    (state: RootState) => state.options.shouldRunESRGAN
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
