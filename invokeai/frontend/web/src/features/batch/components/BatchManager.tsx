import { Flex, Heading, Spacer } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import IAISwitch from 'common/components/IAISwitch';
import {
  asInitialImageToggled,
  batchReset,
  isEnabledChanged,
} from 'features/batch/store/batchSlice';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import BatchImageContainer from './BatchImageGrid';
import { map } from 'lodash-es';
import BatchControlNet from './BatchControlNet';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { controlNets } = state.controlNet;
    const {
      imageNames,
      asInitialImage,
      controlNets: batchControlNets,
      isEnabled,
    } = state.batch;

    return {
      imageCount: imageNames.length,
      asInitialImage,
      controlNets,
      batchControlNets,
      isEnabled,
    };
  },
  defaultSelectorOptions
);

const BatchManager = () => {
  const dispatch = useAppDispatch();
  const { imageCount, isEnabled, controlNets, batchControlNets } =
    useAppSelector(selector);

  const handleResetBatch = useCallback(() => {
    dispatch(batchReset());
  }, [dispatch]);

  const handleToggle = useCallback(() => {
    dispatch(isEnabledChanged(!isEnabled));
  }, [dispatch, isEnabled]);

  const handleChangeAsInitialImage = useCallback(() => {
    dispatch(asInitialImageToggled());
  }, [dispatch]);

  return (
    <Flex
      sx={{
        h: 'full',
        w: 'full',
        flexDir: 'column',
        position: 'relative',
        gap: 2,
        minW: 0,
      }}
    >
      <Flex sx={{ alignItems: 'center' }}>
        <Heading
          size={'md'}
          sx={{ color: 'base.800', _dark: { color: 'base.200' } }}
        >
          {imageCount || 'No'} images
        </Heading>
        <Spacer />
        <IAIButton onClick={handleResetBatch}>Reset</IAIButton>
      </Flex>
      <Flex
        sx={{
          alignItems: 'center',
          flexDir: 'column',
          gap: 4,
        }}
      >
        <IAISwitch
          label="Use as Initial Image"
          onChange={handleChangeAsInitialImage}
        />
        {map(controlNets, (controlNet) => {
          return (
            <BatchControlNet
              key={controlNet.controlNetId}
              controlNet={controlNet}
            />
          );
        })}
      </Flex>
      <BatchImageContainer />
    </Flex>
  );
};

export default BatchManager;
