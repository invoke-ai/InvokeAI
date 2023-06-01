import { Flex, useDisclosure } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import IAICollapse from 'common/components/IAICollapse';
import { memo, useCallback } from 'react';
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

const selector = createSelector(
  controlNetSelector,
  (controlNet) => {
    const { controlNets, isEnabled } = controlNet;

    return { controlNets, isEnabled };
  },
  defaultSelectorOptions
);

const ParamControlNetCollapse = () => {
  const { t } = useTranslation();
  const { controlNets, isEnabled } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const handleClickControlNetToggle = useCallback(() => {
    dispatch(isControlNetEnabledToggled());
  }, [dispatch]);

  const handleClickedAddControlNet = useCallback(() => {
    dispatch(controlNetAdded({ controlNetId: uuidv4() }));
  }, [dispatch]);

  return (
    <IAICollapse
      label={'ControlNet'}
      isOpen={isEnabled}
      onToggle={handleClickControlNetToggle}
      withSwitch
    >
      <Flex sx={{ alignItems: 'flex-end' }}>
        <IAIIconButton
          size="sm"
          aria-label="Add ControlNet"
          onClick={handleClickedAddControlNet}
          icon={<FaPlus />}
        />
      </Flex>
      {map(controlNets, (c) => (
        <ControlNet key={c.controlNetId} controlNet={c} />
      ))}
    </IAICollapse>
  );
};

export default memo(ParamControlNetCollapse);
