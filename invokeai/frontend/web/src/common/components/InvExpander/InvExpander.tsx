import { Divider, Flex } from '@chakra-ui/layout';
import type { SystemStyleObject } from '@chakra-ui/react';
import { Collapse, Icon, useDisclosure } from '@chakra-ui/react';
import type { InvExpanderProps } from 'common/components/InvExpander/types';
import { InvText } from 'common/components/InvText/wrapper';
import { t } from 'i18next';
import { useMemo } from 'react';
import { BiCollapseVertical, BiExpandVertical } from 'react-icons/bi';

export const InvExpander = ({
  children,
  label = t('common.advancedOptions'),
  defaultIsOpen = false,
}: InvExpanderProps) => {
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen });
  const buttonStyles = useMemo<SystemStyleObject>(
    () => ({
      color: isOpen ? 'base.300' : 'base.400',
      borderColor: isOpen ? 'base.300' : 'base.400',
      transitionDuration: 'normal',
      transitionProperty: 'common',
      ':hover, :hover *': {
        transitionDuration: 'normal',
        transitionProperty: 'common',
        color: isOpen ? 'base.200' : 'base.300',
        borderColor: isOpen ? 'base.200' : 'base.300',
      },
    }),
    [isOpen]
  );
  return (
    <Flex flexDir="column" w="full">
      <Flex
        as="button"
        flexDir="row"
        alignItems="center"
        gap={3}
        py={4}
        px={2}
        onClick={onToggle}
        sx={buttonStyles}
      >
        <Divider w="unset" flexGrow={1} sx={buttonStyles} />
        <Flex flexDir="row" alignItems="center" gap={2}>
          <Icon
            as={isOpen ? BiCollapseVertical : BiExpandVertical}
            fontSize="14px"
            sx={buttonStyles}
          />
          <InvText
            variant="subtext"
            fontSize="sm"
            fontWeight="semibold"
            flexShrink={0}
            sx={buttonStyles}
          >
            {label}
          </InvText>
        </Flex>
      </Flex>
      <Collapse in={isOpen} animateOpacity>
        {children}
      </Collapse>
    </Flex>
  );
};
