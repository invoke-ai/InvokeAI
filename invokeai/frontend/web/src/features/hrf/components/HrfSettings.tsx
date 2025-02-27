import { FormControlGroup } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectHrfEnabled } from 'features/hrf/store/hrfSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';

import ParamHrfMethod from './ParamHrfMethod';
import ParamHrfStrength from './ParamHrfStrength';
import ParamHrfToggle from './ParamHrfToggle';

export const HrfSettings = memo(() => {
  const isHRFFeatureEnabled = useFeatureStatus('hrf');
  const hrfEnabled = useAppSelector(selectHrfEnabled);

  if (!isHRFFeatureEnabled) {
    return null;
  }

  return (
    <>
      <ParamHrfToggle />
      <FormControlGroup isDisabled={!hrfEnabled}>
        <ParamHrfStrength />
        <ParamHrfMethod />
      </FormControlGroup>
    </>
  );
});

HrfSettings.displayName = 'HrfSettings';
