import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISelect from 'common/components/IAISelect';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { setIsMaskEnabled, setLayer } from 'features/canvas/store/canvasSlice';
import {
  CanvasLayer,
  LAYER_NAMES_DICT,
} from 'features/canvas/store/canvasTypes';
import { isEqual } from 'lodash-es';

import { ChangeEvent } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const { layer, isMaskEnabled } = canvas;
    return { layer, isMaskEnabled, isStaging };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export default function UnifiedCanvasLayerSelect() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { layer, isMaskEnabled, isStaging } = useAppSelector(selector);

  const handleToggleMaskLayer = () => {
    dispatch(setLayer(layer === 'mask' ? 'base' : 'mask'));
  };

  useHotkeys(
    ['q'],
    () => {
      handleToggleMaskLayer();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [layer]
  );

  const handleChangeLayer = (e: ChangeEvent<HTMLSelectElement>) => {
    const newLayer = e.target.value as CanvasLayer;
    dispatch(setLayer(newLayer));
    if (newLayer === 'mask' && !isMaskEnabled) {
      dispatch(setIsMaskEnabled(true));
    }
  };
  return (
    <IAISelect
      tooltip={`${t('unifiedCanvas.layer')} (Q)`}
      aria-label={`${t('unifiedCanvas.layer')} (Q)`}
      tooltipProps={{ hasArrow: true, placement: 'top' }}
      value={layer}
      validValues={LAYER_NAMES_DICT}
      onChange={handleChangeLayer}
      isDisabled={isStaging}
    />
  );
}
