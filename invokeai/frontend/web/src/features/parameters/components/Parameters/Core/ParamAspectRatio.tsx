import { ButtonGroup, Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { setAspectRatio } from 'features/ui/store/uiSlice';

const aspectRatios = [
  { name: 'Free', value: null },
  { name: '4:3', value: 4 / 3 },
  { name: '16:9', value: 16 / 9 },
  { name: '3:2', value: 3 / 2 },
];

export default function ParamAspectRatio() {
  const aspectRatio = useAppSelector(
    (state: RootState) => state.ui.aspectRatio
  );

  const dispatch = useAppDispatch();

  return (
    <Flex gap={2} flexGrow={1}>
      <ButtonGroup isAttached>
        {aspectRatios.map((ratio) => (
          <IAIButton
            key={ratio.name}
            size="sm"
            isChecked={aspectRatio === ratio.value}
            onClick={() => dispatch(setAspectRatio(ratio.value))}
          >
            {ratio.name}
          </IAIButton>
        ))}
      </ButtonGroup>
    </Flex>
  );
}
