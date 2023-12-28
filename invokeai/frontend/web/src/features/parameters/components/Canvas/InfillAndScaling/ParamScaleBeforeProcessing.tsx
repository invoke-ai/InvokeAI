import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { setBoundingBoxScaleMethod } from 'features/canvas/store/canvasSlice';
import { isBoundingBoxScaleMethod } from 'features/canvas/store/canvasTypes';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const OPTIONS: InvSelectOption[] = [
  { label: 'None', value: 'none' },
  { label: 'Auto', value: 'auto' },
  { label: 'Manual', value: 'manual' },
];

const ParamScaleBeforeProcessing = () => {
  const dispatch = useAppDispatch();
  const boundingBoxScaleMethod = useAppSelector(
    (state) => state.canvas.boundingBoxScaleMethod
  );

  const { t } = useTranslation();

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isBoundingBoxScaleMethod(v?.value)) {
        return;
      }
      dispatch(setBoundingBoxScaleMethod(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => OPTIONS.find((o) => o.value === boundingBoxScaleMethod),
    [boundingBoxScaleMethod]
  );

  return (
    <InvControl
      label={t('parameters.scaleBeforeProcessing')}
      feature="scaleBeforeProcessing"
    >
      <InvSelect value={value} options={OPTIONS} onChange={onChange} />
    </InvControl>
  );
};

export default memo(ParamScaleBeforeProcessing);
