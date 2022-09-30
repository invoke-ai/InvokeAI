import React, { ChangeEvent } from 'react';
import { HEIGHTS } from '../../../app/constants';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import SDSelect from '../../../common/components/SDSelect';
import { setHeight } from '../optionsSlice';
import { fontSize } from './MainOptions';

export default function MainHeight() {
  const height = useAppSelector((state: RootState) => state.options.height);
  const dispatch = useAppDispatch();

  const handleChangeHeight = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setHeight(Number(e.target.value)));

  return (
    <SDSelect
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
