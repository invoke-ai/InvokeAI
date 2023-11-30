import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { setBoundingBoxScaleMethod } from 'features/canvas/store/canvasSlice';
import {
  BOUNDING_BOX_SCALES_DICT,
  BoundingBoxScale,
} from 'features/canvas/store/canvasTypes';

import { memo, useCallback } from 'react';
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
