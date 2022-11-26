import React from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAINumberInput from 'common/components/IAINumberInput';
import { setCfgScale } from 'features/options/store/optionsSlice';
import { inputWidth } from './MainOptions';

export default function MainCFGScale() {
  const dispatch = useAppDispatch();
  const cfgScale = useAppSelector((state: RootState) => state.options.cfgScale);

  const handleChangeCfgScale = (v: number) => dispatch(setCfgScale(v));

  return (
    <IAINumberInput
      label="CFG Scale"
      step={0.5}
      min={1.01}
      max={200}
      onChange={handleChangeCfgScale}
      value={cfgScale}
      width={inputWidth}
      styleClass="main-option-block"
      textAlign="center"
      isInteger={false}
    />
  );
}
