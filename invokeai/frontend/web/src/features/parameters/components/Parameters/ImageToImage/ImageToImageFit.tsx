import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldFitToWidthHeight } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function ImageToImageFit() {
  const dispatch = useAppDispatch();

  const shouldFitToWidthHeight = useAppSelector(
    (state: RootState) => state.generation.shouldFitToWidthHeight
  );

  const handleChangeFit = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldFitToWidthHeight(e.target.checked));

  const { t } = useTranslation();

  return (
    <IAISwitch
      label={t('parameters.imageFit')}
      isChecked={shouldFitToWidthHeight}
      onChange={handleChangeFit}
    />
  );
}
