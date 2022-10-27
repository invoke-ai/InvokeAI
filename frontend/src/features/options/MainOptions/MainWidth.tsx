import React, { ChangeEvent } from 'react';
import { WIDTHS } from '../../../app/constants';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAISelect from '../../../common/components/IAISelect';
import { tabMap } from '../../tabs/InvokeTabs';
import { setWidth } from '../optionsSlice';
import { fontSize } from './MainOptions';

export default function MainWidth() {
  const { width, activeTab } = useAppSelector(
    (state: RootState) => state.options
  );
  const dispatch = useAppDispatch();

  const handleChangeWidth = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setWidth(Number(e.target.value)));

  return (
    <IAISelect
      isDisabled={tabMap[activeTab] === 'inpainting'}
      label="Width"
      value={width}
      flexGrow={1}
      onChange={handleChangeWidth}
      validValues={WIDTHS}
      fontSize={fontSize}
      styleClass="main-option-block"
    />
  );
}
