import { useAppSelector } from 'app/store/storeHooks';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';

import ParamHrfMethod from './ParamHrfMethod';
import ParamHrfStrength from './ParamHrfStrength';
import ParamHrfToggle from './ParamHrfToggle';

export const HrfSettings = memo(() => {
  const isHRFFeatureEnabled = useFeatureStatus('hrf').isFeatureEnabled;
  const hrfEnabled = useAppSelector((s) => s.hrf.hrfEnabled);

  if (!isHRFFeatureEnabled) {
    return null;
  }

  return (
    <>
      <ParamHrfToggle />
      <InvControlGroup isDisabled={!hrfEnabled}>
        <ParamHrfStrength />
        <ParamHrfMethod />
      </InvControlGroup>
    </>
  );
});

HrfSettings.displayName = 'HrfSettings';
