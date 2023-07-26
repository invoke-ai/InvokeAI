import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  ResizeModes,
  controlNetResizeModeChanged,
} from 'features/controlNet/store/controlNetSlice';
import { useCallback, useMemo } from 'react';

type ParamControlNetResizeModeProps = {
  controlNetId: string;
};

const RESIZE_MODE_DATA = [
  { label: 'Resize', value: 'just_resize' },
  { label: 'Crop', value: 'crop_resize' },
  { label: 'Fill', value: 'fill_resize' },
];

export default function ParamControlNetResizeMode(
  props: ParamControlNetResizeModeProps
) {
  const { controlNetId } = props;
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlNet }) => {
          const { resizeMode, isEnabled } =
            controlNet.controlNets[controlNetId];
          return { resizeMode, isEnabled };
        },
        defaultSelectorOptions
      ),
    [controlNetId]
  );

  const { resizeMode, isEnabled } = useAppSelector(selector);

  const handleResizeModeChange = useCallback(
    (resizeMode: ResizeModes) => {
      dispatch(controlNetResizeModeChanged({ controlNetId, resizeMode }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAIMantineSelect
      disabled={!isEnabled}
      label="Resize Mode"
      data={RESIZE_MODE_DATA}
      value={String(resizeMode)}
      onChange={handleResizeModeChange}
    />
  );
}
