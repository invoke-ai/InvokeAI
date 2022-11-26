import React, { ChangeEvent } from 'react';
import { HEIGHTS } from 'app/constants';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAISelect from 'common/components/IAISelect';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { setHeight } from 'features/options/store/optionsSlice';

export default function MainHeight() {
  const height = useAppSelector((state: RootState) => state.options.height);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const dispatch = useAppDispatch();

  const handleChangeHeight = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setHeight(Number(e.target.value)));

  return (
    <IAISelect
      isDisabled={activeTabName === 'unifiedCanvas'}
      label="Height"
      value={height}
      flexGrow={1}
      onChange={handleChangeHeight}
      validValues={HEIGHTS}
      styleClass="main-option-block"
    />
  );
}
