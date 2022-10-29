import React, { ChangeEvent } from 'react';
import { HEIGHTS } from '../../../app/constants';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAISelect from '../../../common/components/IAISelect';
import { tabMap } from '../../tabs/InvokeTabs';
import { setHeight } from '../optionsSlice';
import { fontSize } from './MainOptions';

export default function MainHeight() {
  const { activeTab, height } = useAppSelector(
    (state: RootState) => state.options
  );
  const dispatch = useAppDispatch();

  const handleChangeHeight = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setHeight(Number(e.target.value)));

  return (
    <IAISelect
      isDisabled={tabMap[activeTab] === 'inpainting'}
      label="Height"
      value={height}
      flexGrow={1}
      onChange={handleChangeHeight}
      validValues={HEIGHTS}
      fontSize={fontSize}
      styleClass="main-option-block"
    />
  );
}
