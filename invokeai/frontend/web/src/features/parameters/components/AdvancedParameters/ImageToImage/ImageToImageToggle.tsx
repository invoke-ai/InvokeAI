import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { isImageToImageEnabledChanged } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function ImageToImageToggle() {
  const isImageToImageEnabled = useAppSelector(
    (state: RootState) => state.generation.isImageToImageEnabled
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const handleChange = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(isImageToImageEnabledChanged(e.target.checked));

  return (
    <IAISwitch
      label={t('common.img2img')}
      isChecked={isImageToImageEnabled}
      width="auto"
      onChange={handleChange}
    />
  );
}
