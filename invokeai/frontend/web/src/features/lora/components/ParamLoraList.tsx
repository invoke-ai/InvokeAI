import { Divider, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { map } from 'lodash-es';
import ParamLora from './ParamLora';
import { memo } from 'react';

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
        <Flex key={lora.model_name} sx={{ flexDirection: 'column', gap: 2 }}>
          {i > 0 && <Divider pt={1} />}
          <ParamLora lora={lora} />
        </Flex>
      ))}
    </>
  );
};

export default memo(ParamLoraList);
