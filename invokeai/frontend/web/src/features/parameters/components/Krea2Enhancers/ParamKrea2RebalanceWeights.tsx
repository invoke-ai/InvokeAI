import { FormControl, FormLabel, Input } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  isValidKrea2RebalanceWeights,
  selectKrea2RebalanceWeights,
  setKrea2RebalanceWeights,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamKrea2RebalanceWeights = () => {
  const { t } = useTranslation();
  const weights = useAppSelector(selectKrea2RebalanceWeights);
  const dispatch = useAppDispatch();

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setKrea2RebalanceWeights(e.target.value));
    },
    [dispatch]
  );

  // Surface the same requirement readiness enforces (exactly 12 finite numbers), so an invalid string that
  // blocks generation is visible in the field rather than only in the queue-blocked reason.
  const isInvalid = !isValidKrea2RebalanceWeights(weights);

  return (
    <FormControl orientation="vertical" isInvalid={isInvalid}>
      <InformationalPopover feature="krea2RebalanceWeights">
        <FormLabel>{t('parameters.krea2RebalanceWeights')}</FormLabel>
      </InformationalPopover>
      <Input value={weights} onChange={onChange} placeholder={t('parameters.krea2RebalanceWeightsPlaceholder')} />
    </FormControl>
  );
};

export default memo(ParamKrea2RebalanceWeights);
