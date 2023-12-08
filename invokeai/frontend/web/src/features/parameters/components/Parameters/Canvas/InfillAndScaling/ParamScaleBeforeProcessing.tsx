import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { setBoundingBoxScaleMethod } from 'features/canvas/store/canvasSlice';
import {
  BOUNDING_BOX_SCALES_DICT,
  BoundingBoxScale,
} from 'features/canvas/store/canvasTypes';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector([stateSelector], ({ canvas }) => {
  const { boundingBoxScaleMethod: boundingBoxScale } = canvas;

  return {
    boundingBoxScale,
  };
});

const ParamScaleBeforeProcessing = () => {
  const dispatch = useAppDispatch();
  const { boundingBoxScale } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleChangeBoundingBoxScaleMethod = useCallback(
    (v: string) => {
      dispatch(setBoundingBoxScaleMethod(v as BoundingBoxScale));
    },
    [dispatch]
  );

  return (
    <IAIInformationalPopover feature="scaleBeforeProcessing">
      <IAIMantineSearchableSelect
        label={t('parameters.scaleBeforeProcessing')}
        data={BOUNDING_BOX_SCALES_DICT}
        value={boundingBoxScale}
        onChange={handleChangeBoundingBoxScaleMethod}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamScaleBeforeProcessing);
