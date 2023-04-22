import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { isImageToImageEnabledChanged } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';

export default function ImageToImageToggle() {
  const isImageToImageEnabled = useAppSelector(
    (state: RootState) => state.generation.isImageToImageEnabled
  );

  const dispatch = useAppDispatch();

  const handleChange = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(isImageToImageEnabledChanged(e.target.checked));

  return (
    <IAISwitch
      isChecked={isImageToImageEnabled}
      width="auto"
      onChange={handleChange}
    />
  );
}
