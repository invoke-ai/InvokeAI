import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setInfillTileSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamInfillTileSize = () => {
  const dispatch = useAppDispatch();
  const infillTileSize = useAppSelector((s) => s.generation.infillTileSize);
  const infillMethod = useAppSelector((s) => s.generation.infillMethod);

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
