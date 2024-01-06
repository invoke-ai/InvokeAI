import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setInfillTileSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector([stateSelector], ({ generation }) => {
  const { infillTileSize, infillMethod } = generation;

  return {
    infillTileSize,
    infillMethod,
  };
});

const ParamInfillTileSize = () => {
  const dispatch = useAppDispatch();
  const { infillTileSize, infillMethod } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setInfillTileSize(v));
    },
    [dispatch]
  );

  return (
    <InvControl
      isDisabled={infillMethod !== 'tile'}
      label={t('parameters.tileSize')}
    >
      <InvSlider
        min={16}
        max={64}
        numberInputMax={256}
        value={infillTileSize}
        defaultValue={32}
        onChange={handleChange}
        withNumberInput
        marks
      />
    </InvControl>
  );
};

export default memo(ParamInfillTileSize);
