import { Divider } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { map } from 'lodash-es';
import ParamLora from './ParamLora';

const selector = createSelector(
  stateSelector,
  ({ lora }) => {
    return { lorasArray: map(lora.loras) };
  },
  defaultSelectorOptions
);

const ParamLoraList = () => {
  const { lorasArray } = useAppSelector(selector);

  return (
    <>
      {lorasArray.map((lora, i) => (
        <>
          {i > 0 && <Divider key={`${lora.model_name}-divider`} pt={1} />}
          <ParamLora key={lora.model_name} lora={lora} />
        </>
      ))}
    </>
  );
};

export default ParamLoraList;
