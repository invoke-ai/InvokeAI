import React from 'react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import { setCfgScale } from 'features/options/store/optionsSlice';

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
      width="auto"
      styleClass="main-option-block"
      textAlign="center"
      isInteger={false}
    />
  );
}
