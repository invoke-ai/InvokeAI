import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { map } from 'lodash-es';
import ParamLora from './ParamLora';

const selector = createSelector(stateSelector, ({ lora }) => {
  const { loras } = lora;

  return { loras };
});

const ParamLoraList = () => {
  const { loras } = useAppSelector(selector);

  return map(loras, (lora) => <ParamLora key={lora.name} lora={lora} />);
};

export default ParamLoraList;
