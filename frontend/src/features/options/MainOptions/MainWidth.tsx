import React, { ChangeEvent } from 'react';
import { WIDTHS } from '../../../app/constants';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import SDSelect from '../../../common/components/SDSelect';
import { setWidth } from '../optionsSlice';
import { fontSize } from './MainOptions';

export default function MainWidth() {
  const width = useAppSelector((state: RootState) => state.options.width);
  const dispatch = useAppDispatch();

  const handleChangeWidth = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setWidth(Number(e.target.value)));

  return (
    <SDSelect
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
