import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISelect from 'common/components/IAISelect';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { setBoundingBoxScaleMethod } from 'features/canvas/store/canvasSlice';
import {
  BoundingBoxScale,
  BOUNDING_BOX_SCALES_DICT,
} from 'features/canvas/store/canvasTypes';

import { ChangeEvent, memo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { boundingBoxScaleMethod: boundingBoxScale } = canvas;

    return {
      boundingBoxScale,
    };
  },
  defaultSelectorOptions
);

const ParamScaleBeforeProcessing = () => {
  const dispatch = useAppDispatch();
  const { boundingBoxScale } = useAppSelector(selector);

  const { t } = useTranslation();

  const handleChangeBoundingBoxScaleMethod = (
    e: ChangeEvent<HTMLSelectElement>
  ) => {
    dispatch(setBoundingBoxScaleMethod(e.target.value as BoundingBoxScale));
  };

  return (
    <IAISelect
      label={t('parameters.scaleBeforeProcessing')}
      validValues={BOUNDING_BOX_SCALES_DICT}
      value={boundingBoxScale}
      onChange={handleChangeBoundingBoxScaleMethod}
    />
  );
};

export default memo(ParamScaleBeforeProcessing);
