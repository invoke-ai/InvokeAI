import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { map } from 'lodash-es';
import ParamLora from './ParamLora';

const selector = createSelector(
  stateSelector,
  ({ lora }) => {
    const { loras } = lora;

    return { loras };
  },
  defaultSelectorOptions
);

const ParamLoraList = () => {
  const { loras } = useAppSelector(selector);

  return (
    <>
      {map(loras, (lora) => (
        <ParamLora key={lora.model_name} lora={lora} />
      ))}
    </>
  );
};

export default ParamLoraList;
