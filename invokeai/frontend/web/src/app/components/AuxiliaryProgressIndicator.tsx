import { Flex, Spinner, Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { memo } from 'react';

const selector = createSelector(systemSelector, (system) => {
  const { isUploading } = system;

  let tooltip = '';

  if (isUploading) {
    tooltip = 'Uploading...';
  }

  return {
    tooltip,
    shouldShow: isUploading,
  };
});

export const AuxiliaryProgressIndicator = () => {
  const { shouldShow, tooltip } = useAppSelector(selector);

  if (!shouldShow) {
    return null;
  }

  return (
    <Flex
      sx={{
        alignItems: 'center',
        justifyContent: 'center',
        color: 'base.600',
      }}
    >
      <Tooltip label={tooltip} placement="right" hasArrow>
        <Spinner />
      </Tooltip>
    </Flex>
  );
};

export default memo(AuxiliaryProgressIndicator);
