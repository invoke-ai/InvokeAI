import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { controlNetAutoConfigToggled } from 'features/controlNet/store/controlNetSlice';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { memo, useCallback, useMemo } from 'react';

type Props = {
  controlNetId: string;
};

const ParamControlNetShouldAutoConfig = (props: Props) => {
  const { controlNetId } = props;
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlNet }) => {
          const { isEnabled, shouldAutoConfig } =
            controlNet.controlNets[controlNetId];
          return { isEnabled, shouldAutoConfig };
        },
        defaultSelectorOptions
      ),
    [controlNetId]
  );

  const { isEnabled, shouldAutoConfig } = useAppSelector(selector);
  const isBusy = useAppSelector(selectIsBusy);

  const handleShouldAutoConfigChanged = useCallback(() => {
    dispatch(controlNetAutoConfigToggled({ controlNetId }));
  }, [controlNetId, dispatch]);

  return (
    <IAISwitch
      label="Auto configure processor"
      aria-label="Auto configure processor"
      isChecked={shouldAutoConfig}
      onChange={handleShouldAutoConfigChanged}
      isDisabled={isBusy || !isEnabled}
    />
  );
};

export default memo(ParamControlNetShouldAutoConfig);
