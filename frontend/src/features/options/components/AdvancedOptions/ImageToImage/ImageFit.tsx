import React, { ChangeEvent } from 'react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldFitToWidthHeight } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

export default function ImageFit() {
  const dispatch = useAppDispatch();

  const shouldFitToWidthHeight = useAppSelector(
    (state: RootState) => state.options.shouldFitToWidthHeight
  );

  const handleChangeFit = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldFitToWidthHeight(e.target.checked));

  const { t } = useTranslation();

  return (
    <IAISwitch
      label={t('options:imageFit')}
      isChecked={shouldFitToWidthHeight}
      onChange={handleChangeFit}
    />
  );
}
