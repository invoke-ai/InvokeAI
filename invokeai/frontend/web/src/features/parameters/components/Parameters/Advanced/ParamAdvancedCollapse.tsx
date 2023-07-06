import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAICollapse from 'common/components/IAICollapse';
import ParamClipSkip from './ParamClipSkip';

export default function ParamAdvancedCollapse() {
  const shouldShowAdvancedOptions = useAppSelector(
    (state: RootState) => state.ui.shouldShowAdvancedOptions
  );
  return (
    shouldShowAdvancedOptions && (
      <IAICollapse label={'Advanced'}>
        <Flex sx={{ flexDir: 'column', gap: 2 }}>
          <ParamClipSkip />
        </Flex>
      </IAICollapse>
    )
  );
}
