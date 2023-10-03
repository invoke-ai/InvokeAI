import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ParamIPAdapterBeginEnd from './ParamIPAdapterBeginEnd';
import ParamIPAdapterFeatureToggle from './ParamIPAdapterFeatureToggle';
import ParamIPAdapterImage from './ParamIPAdapterImage';
import ParamIPAdapterModelSelect from './ParamIPAdapterModelSelect';
import ParamIPAdapterWeight from './ParamIPAdapterWeight';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from '../../../../app/store/store';
import { defaultSelectorOptions } from '../../../../app/store/util/defaultMemoizeOptions';
import { useAppSelector } from '../../../../app/store/storeHooks';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { isIPAdapterEnabled } = state.controlNet;

    return { isIPAdapterEnabled };
  },
  defaultSelectorOptions
);

const IPAdapterPanel = () => {
  const { isIPAdapterEnabled } = useAppSelector(selector);
  return (
    <Flex
      sx={{
        flexDir: 'column',
        gap: 3,
        paddingInline: 3,
        paddingBlock: 2,
        borderRadius: 'base',
        position: 'relative',
        bg: 'base.250',
        _dark: {
          bg: 'base.750',
        },
      }}
    >
      <ParamIPAdapterFeatureToggle />
      {isIPAdapterEnabled && (
        <>
          <ParamIPAdapterModelSelect />
          <Flex gap="3">
            <Flex
              flexDirection="column"
              sx={{
                h: 28,
                w: 'full',
                gap: 4,
                mb: 4,
              }}
            >
              <ParamIPAdapterWeight />
              <ParamIPAdapterBeginEnd />
            </Flex>
            <ParamIPAdapterImage />
          </Flex>
        </>
      )}
    </Flex>
  );
};

export default memo(IPAdapterPanel);
