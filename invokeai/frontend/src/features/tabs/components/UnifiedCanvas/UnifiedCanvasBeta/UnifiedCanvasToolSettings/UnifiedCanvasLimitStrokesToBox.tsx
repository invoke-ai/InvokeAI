import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldRestrictStrokesToBox } from 'features/canvas/store/canvasSlice';
import React from 'react';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasLimitStrokesToBox() {
  const dispatch = useAppDispatch();

  const shouldRestrictStrokesToBox = useAppSelector(
    (state: RootState) => state.canvas.shouldRestrictStrokesToBox
  );

  const { t } = useTranslation();

  return (
    <IAICheckbox
      label={t('unifiedcanvas:betaLimitToBox')}
      isChecked={shouldRestrictStrokesToBox}
      onChange={(e) =>
        dispatch(setShouldRestrictStrokesToBox(e.target.checked))
      }
    />
  );
}
