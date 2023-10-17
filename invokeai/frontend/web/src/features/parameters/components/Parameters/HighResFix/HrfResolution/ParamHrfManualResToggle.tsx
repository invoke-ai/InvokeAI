import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHrfManualResEnabled } from 'features/parameters/store/generationSlice';
import { ChangeEvent, useCallback, useEffect } from 'react';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import {
  setHrfHeight,
  setHrfWidth,
} from 'features/parameters/store/generationSlice';

export const calculateHrfRes = (
  dispatch: ReturnType<typeof useAppDispatch>,
  width: number,
  height: number,
  aspectRatio: number | null
) => {
  const newWidth = roundToMultiple((width * 2) / 3, 8);
  let newHeight = roundToMultiple((height * 2) / 3, 8);
  if (aspectRatio) {
    newHeight = roundToMultiple(newWidth / aspectRatio, 8);
  }
  dispatch(setHrfWidth(newWidth));
  dispatch(setHrfHeight(newHeight));
};
// Toggle whether to let the user set their own initial resolution or calculate it
// automatically.
export default function ParamHrfAutoRes() {
  const dispatch = useAppDispatch();
  const tooltip =
    'When not specified, the initial height and width are determined automatically.';

  const hrfManualResEnabled = useAppSelector(
    (state: RootState) => state.generation.hrfManualResEnabled
  );
  const width = useAppSelector((state: RootState) => state.generation.width);
  const height = useAppSelector((state: RootState) => state.generation.height);
  const aspectRatio = useAppSelector(
    (state: RootState) => state.generation.aspectRatio
  );

  useEffect(() => {
    if (!hrfManualResEnabled) {
      calculateHrfRes(dispatch, width, height, aspectRatio);
    }
  }, [dispatch, hrfManualResEnabled, width, height, aspectRatio]);

  const handleHrfManualResEnabled = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setHrfManualResEnabled(e.target.checked));

      if (!e.target.checked) {
        calculateHrfRes(dispatch, width, height, aspectRatio);
      }
    },
    [dispatch, width, height, aspectRatio]
  );

  return (
    <IAISwitch
      label="Manual Initial Resolution"
      isChecked={hrfManualResEnabled}
      onChange={handleHrfManualResEnabled}
      tooltip={tooltip}
    />
  );
}
