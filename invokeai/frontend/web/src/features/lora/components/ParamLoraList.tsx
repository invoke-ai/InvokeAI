import { Divider, Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { map } from 'lodash-es';
import { memo } from 'react';
import ParamLora from './ParamLora';

const selector = createMemoizedSelector(stateSelector, ({ lora }) => {
  return { lorasArray: map(lora.loras) };
});

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
