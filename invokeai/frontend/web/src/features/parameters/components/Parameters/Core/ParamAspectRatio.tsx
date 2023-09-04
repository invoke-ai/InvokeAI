import { ButtonGroup, Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import {
  setAspectRatio,
  setShouldLockAspectRatio,
} from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from '../../../../ui/store/uiSelectors';

const aspectRatios = [
  { name: 'Free', value: null },
  { name: '2:3', value: 2 / 3 },
  { name: '16:9', value: 16 / 9 },
  { name: '1:1', value: 1 / 1 },
];

export const mappedAspectRatios = aspectRatios.map((ar) => ar.value);

export default function ParamAspectRatio() {
  const aspectRatio = useAppSelector(
    (state: RootState) => state.generation.aspectRatio
  );

  const dispatch = useAppDispatch();
  const shouldFitToWidthHeight = useAppSelector(
    (state: RootState) => state.generation.shouldFitToWidthHeight
  );
  const activeTabName = useAppSelector(activeTabNameSelector);

  return (
    <Flex gap={2} flexGrow={1}>
      <ButtonGroup isAttached>
        {aspectRatios.map((ratio) => (
          <IAIButton
            key={ratio.name}
            size="sm"
            isChecked={aspectRatio === ratio.value}
            isDisabled={
              activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
            }
            onClick={() => {
              dispatch(setAspectRatio(ratio.value));
              dispatch(setShouldLockAspectRatio(false));
            }}
          >
            {ratio.name}
          </IAIButton>
        ))}
      </ButtonGroup>
    </Flex>
  );
}
