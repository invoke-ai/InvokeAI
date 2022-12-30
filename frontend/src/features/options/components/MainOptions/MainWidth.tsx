import React, { ChangeEvent } from 'react';
import { WIDTHS } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { setWidth } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

export default function MainWidth() {
  const width = useAppSelector((state: RootState) => state.options.width);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const handleChangeWidth = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setWidth(Number(e.target.value)));

  return (
    <IAISelect
      isDisabled={activeTabName === 'unifiedCanvas'}
      label={t('options:width')}
      value={width}
      flexGrow={1}
      onChange={handleChangeWidth}
      validValues={WIDTHS}
      styleClass="main-option-block"
    />
  );
}
