import React from 'react';
import { useDispatch } from 'react-redux';
import { RootState, useAppSelector } from '../../../app/store';
import SDNumberInput from '../../../common/components/SDNumberInput';
import { setCfgScale } from '../optionsSlice';
import { fontSize, inputWidth } from './MainOptions';

export default function MainCFGScale() {
  const dispatch = useDispatch();
  const cfgScale = useAppSelector((state: RootState) => state.options.cfgScale);

  const handleChangeCfgScale = (v: string | number) =>
    dispatch(setCfgScale(Number(v)));

  return (
    <SDNumberInput
      label="CFG Scale"
      step={0.5}
      min={1}
      max={30}
      onChange={handleChangeCfgScale}
      value={cfgScale}
      width={inputWidth}
      fontSize={fontSize}
      styleClass="main-option-block"
      textAlign="center"
    />
  );
}
