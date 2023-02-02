import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAICheckbox from 'common/components/IAICheckbox';
import { setShouldPreserveMaskedArea } from 'features/canvas/store/canvasSlice';
import React from 'react';
import { useTranslation } from 'react-i18next';

export default function UnifiedCanvasPreserveMask() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldPreserveMaskedArea = useAppSelector(
    (state: RootState) => state.canvas.shouldPreserveMaskedArea
  );

  return (
    <IAICheckbox
      label={t('unifiedcanvas:betaPreserveMasked')}
      isChecked={shouldPreserveMaskedArea}
      onChange={(e) => dispatch(setShouldPreserveMaskedArea(e.target.checked))}
    />
  );
}
