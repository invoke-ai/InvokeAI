import { Flex } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';
import { memo } from 'react';

import type { InvSelectFallbackProps } from './types';

export const InvSelectFallback = memo((props: InvSelectFallbackProps) => (
  <Flex h={8} alignItems="center" justifyContent="center">
    <InvText fontSize="sm" color="base.500">
      {props.label}
    </InvText>
  </Flex>
));

InvSelectFallback.displayName = 'InvSelectFallback';
