import {
  AccordionButton as ChakraAccordionButton,
  Spacer,
} from '@chakra-ui/react';
import { InvBadge } from 'common/components/InvBadge/wrapper';
import { truncate } from 'lodash-es';
import { useMemo } from 'react';

import type { InvAccordionButtonProps } from './types';
import { InvAccordionIcon } from './wrapper';

export const InvAccordionButton = (props: InvAccordionButtonProps) => {
  const { children, badges: _badges, ...rest } = props;
  const badges = useMemo<string[] | undefined>(
    () =>
      _badges?.map((b) => truncate(String(b), { length: 24, omission: '...' })),
    [_badges]
  );
  return (
    <ChakraAccordionButton {...rest}>
      {children}
      <Spacer />
      {badges?.map((b, i) => (
        <InvBadge key={`${b}.${i}`} colorScheme="blue">
          {b}
        </InvBadge>
      ))}
      <InvAccordionIcon />
    </ChakraAccordionButton>
  );
};
