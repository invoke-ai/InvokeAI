import { ButtonGroup } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import {
  setAspectRatio,
  setShouldLockAspectRatio,
} from 'features/parameters/store/generationSlice';
import i18next from 'i18next';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';

const aspectRatios = [
  { name: i18next.t('parameters.aspectRatioFree'), value: null },
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

  const handleClick = useCallback(
    (ratio: (typeof aspectRatios)[number]) => {
      dispatch(setAspectRatio(ratio.value));
      dispatch(setShouldLockAspectRatio(false));
    },
    [dispatch]
  );

  return (
    <ButtonGroup isAttached>
      {aspectRatios.map((ratio) => (
        <IAIButton
          key={ratio.name}
          size="sm"
          isChecked={aspectRatio === ratio.value}
          isDisabled={
            activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
          }
          onClick={handleClick.bind(null, ratio)}
        >
          {ratio.name}
        </IAIButton>
      ))}
    </ButtonGroup>
  );
}
