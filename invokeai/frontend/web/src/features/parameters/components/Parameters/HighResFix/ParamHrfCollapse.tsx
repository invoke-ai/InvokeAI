import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { useMemo } from 'react';
import ParamHrfScale from './ParamHrfScale';
import ParamHrfStrength from './ParamHrfStrength';
import ParamHrfToggle from './ParamHrfToggle';

const selector = createSelector(
  stateSelector,
  (state: RootState) => {
    const { hrfToggled } = state.generation;

    return { hrfToggled };
  },
  defaultSelectorOptions
);

export default function ParamHrfCollapse() {
  const { hrfToggled } = useAppSelector(selector);
  const activeLabel = useMemo(() => {
    if (hrfToggled) {
      return 'On';
    } else {
      return 'Off';
    }
  }, [hrfToggled]);

  return (
    <IAICollapse label="High Resolution Fix" activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamHrfToggle />
        <ParamHrfScale />
        <ParamHrfStrength />
      </Flex>
    </IAICollapse>
  );
}
