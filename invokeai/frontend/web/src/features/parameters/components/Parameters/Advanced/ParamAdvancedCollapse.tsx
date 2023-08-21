import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import ParamClipSkip from './ParamClipSkip';

const selector = createSelector(
  stateSelector,
  (state: RootState) => {
    const clipSkip = state.generation.clipSkip;
    return {
      activeLabel: clipSkip > 0 ? 'Clip Skip' : undefined,
    };
  },
  defaultSelectorOptions
);

export default function ParamAdvancedCollapse() {
  const { activeLabel } = useAppSelector(selector);
  const shouldShowAdvancedOptions = useAppSelector(
    (state: RootState) => state.generation.shouldShowAdvancedOptions
  );

  if (!shouldShowAdvancedOptions) {
    return null;
  }

  return (
    <IAICollapse label="Advanced" activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamClipSkip />
      </Flex>
    </IAICollapse>
  );
}
