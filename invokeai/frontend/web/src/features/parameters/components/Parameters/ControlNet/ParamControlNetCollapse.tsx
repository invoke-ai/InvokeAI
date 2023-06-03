import {
  Divider,
  Flex,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import IAICollapse from 'common/components/IAICollapse';
import { Fragment, memo, useCallback } from 'react';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaPlus } from 'react-icons/fa';
import ControlNet from 'features/controlNet/components/ControlNet';
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
import ControlNetMini from 'features/controlNet/components/ControlNetMini';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import IAIButton from 'common/components/IAIButton';

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
      {controlNetsArray.length === 0 && (
        <IAIButton onClick={handleClickedAddControlNet}>
          Add ControlNet
        </IAIButton>
      )}
      <Flex sx={{ flexDir: 'column', gap: 4 }}>
        {controlNetsArray.map((c, i) => (
          <Fragment key={c.controlNetId}>
            {i > 0 && <Divider />}
            <ControlNetMini controlNet={c} />
          </Fragment>
        ))}
      </Flex>
    </IAICollapse>
  );

  return (
    <IAICollapse
      label={'ControlNet'}
      isOpen={isEnabled}
      onToggle={handleClickControlNetToggle}
      withSwitch
    >
      <Tabs
        isFitted
        orientation="horizontal"
        variant="line"
        size="sm"
        colorScheme="accent"
      >
        <TabList alignItems="center" borderBottomColor="base.800" pb={4}>
          {controlNetsArray.map((c, i) => (
            <Tab key={`tab_${c.controlNetId}`} borderTopRadius="base">
              {i + 1}
            </Tab>
          ))}
          <IAIIconButton
            marginInlineStart={2}
            size="sm"
            aria-label="Add ControlNet"
            onClick={handleClickedAddControlNet}
            icon={<FaPlus />}
          />
        </TabList>
        <TabPanels>
          {controlNetsArray.map((c) => (
            <TabPanel key={`tabPanel_${c.controlNetId}`} sx={{ p: 0 }}>
              <ControlNet controlNet={c} />
              {/* <ControlNetMini controlNet={c} /> */}
            </TabPanel>
          ))}
        </TabPanels>
      </Tabs>
    </IAICollapse>
  );
};

export default memo(ParamControlNetCollapse);
