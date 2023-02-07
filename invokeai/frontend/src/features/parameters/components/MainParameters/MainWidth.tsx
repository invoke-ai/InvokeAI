import { WIDTHS } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { setWidth } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function MainWidth() {
  const width = useAppSelector((state: RootState) => state.generation.width);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const handleChangeWidth = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setWidth(Number(e.target.value)));

  return (
    <IAISelect
      isDisabled={activeTabName === 'unifiedCanvas'}
      label={t('parameters:width')}
      value={width}
      flexGrow={1}
      onChange={handleChangeWidth}
      validValues={WIDTHS}
      styleClass="main-settings-block"
    />
  );
}
