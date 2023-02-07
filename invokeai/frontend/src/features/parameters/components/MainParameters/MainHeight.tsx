import { HEIGHTS } from 'app/constants';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { setHeight } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function MainHeight() {
  const height = useAppSelector((state: RootState) => state.generation.height);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChangeHeight = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setHeight(Number(e.target.value)));

  return (
    <IAISelect
      isDisabled={activeTabName === 'unifiedCanvas'}
      label={t('parameters:height')}
      value={height}
      flexGrow={1}
      onChange={handleChangeHeight}
      validValues={HEIGHTS}
      styleClass="main-settings-block"
    />
  );
}
