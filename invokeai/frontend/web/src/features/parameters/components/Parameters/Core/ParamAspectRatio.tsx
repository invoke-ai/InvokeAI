import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { setAspectRatio } from 'features/ui/store/uiSlice';
import { ReactNode } from 'react';

const aspectRatios = [
  { name: 'Free', value: null },
  { name: '4:3', value: 4 / 3 },
  { name: '16:9', value: 16 / 9 },
  { name: '3:2', value: 3 / 2 },
];

export const roundToEight = (number: number) => {
  return Math.round(number / 8) * 8;
};

export default function ParamAspectRatio() {
  const aspectRatio = useAppSelector(
    (state: RootState) => state.ui.aspectRatio
  );

  const dispatch = useAppDispatch();

  const renderAspectRatios = () => {
    const aspectRatiosToRender: ReactNode[] = [];
    aspectRatios.forEach((ratio) => {
      aspectRatiosToRender.push(
        <IAIButton
          key={ratio.name}
          size="sm"
          width="max-content"
          isChecked={aspectRatio === ratio.value}
          onClick={() => dispatch(setAspectRatio(ratio.value))}
        >
          {ratio.name}
        </IAIButton>
      );
    });
    return aspectRatiosToRender;
  };

  return (
    <Flex gap={2} w="100%">
      {renderAspectRatios()}
    </Flex>
  );
}
