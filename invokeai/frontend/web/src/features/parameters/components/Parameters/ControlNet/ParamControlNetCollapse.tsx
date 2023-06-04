import { Divider, Flex } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import IAICollapse from 'common/components/IAICollapse';
import { Fragment, memo, useCallback } from 'react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { createSelector } from '@reduxjs/toolkit';
import {
  controlNetAdded,
  controlNetSelector,
  isControlNetEnabledToggled,
} from 'features/controlNet/store/controlNetSlice';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { map } from 'lodash-es';
import { v4 as uuidv4 } from 'uuid';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import IAIButton from 'common/components/IAIButton';
import ControlNet from 'features/controlNet/components/ControlNet';

const selector = createSelector(
  controlNetSelector,
  (controlNet) => {
    const { controlNets, isEnabled } = controlNet;

    return { controlNetsArray: map(controlNets), isEnabled };
  },
  defaultSelectorOptions
);

const ParamControlNetCollapse = () => {
  const { t } = useTranslation();
  const { controlNetsArray, isEnabled } = useAppSelector(selector);
  const isControlNetDisabled = useFeatureStatus('controlNet').isFeatureDisabled;
  const dispatch = useAppDispatch();

  const handleClickControlNetToggle = useCallback(() => {
    dispatch(isControlNetEnabledToggled());
  }, [dispatch]);

  const handleClickedAddControlNet = useCallback(() => {
    dispatch(controlNetAdded({ controlNetId: uuidv4() }));
  }, [dispatch]);

  if (isControlNetDisabled) {
    return null;
  }

  return (
    <IAICollapse
      label={'ControlNet'}
      isOpen={isEnabled}
      onToggle={handleClickControlNetToggle}
      withSwitch
    >
      <Flex sx={{ flexDir: 'column', gap: 3 }}>
        {controlNetsArray.map((c, i) => (
          <Fragment key={c.controlNetId}>
            {i > 0 && <Divider />}
            <ControlNet controlNet={c} />
          </Fragment>
        ))}
        <IAIButton flexGrow={1} onClick={handleClickedAddControlNet}>
          Add ControlNet
        </IAIButton>
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamControlNetCollapse);
