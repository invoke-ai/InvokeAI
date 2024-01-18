import { Divider, Flex } from '@chakra-ui/layout';
import type { SystemStyleObject } from '@chakra-ui/react';
import { Collapse, Icon, useDisclosure } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import type { InvExpanderProps } from 'common/components/InvExpander/types';
import { InvText } from 'common/components/InvText/wrapper';
import { expanderToggled } from 'features/parameters/store/actions';
import { t } from 'i18next';
import { useCallback } from 'react';
import { BiCollapseVertical, BiExpandVertical } from 'react-icons/bi';

const buttonStyles: SystemStyleObject = {
  color: 'base.400',
  borderColor: 'base.400',
  transitionDuration: 'normal',
  transitionProperty: 'common',
  ':hover, :hover *': {
    transitionDuration: 'normal',
    transitionProperty: 'common',
    color: 'base.300',
    borderColor: 'base.300',
  },
};

export const InvExpander = ({
  children,
  label = t('common.advancedOptions'),
  defaultIsOpen = false,
  id,
}: InvExpanderProps) => {
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen });
  const dispatch = useAppDispatch();

  const onToggleExpander = useCallback(() => {
    if (id) {
      dispatch(expanderToggled({ id, isOpen }));
    }

    onToggle();
  }, [id, dispatch, onToggle, isOpen]);

  return (
    <Flex flexDir="column" w="full">
      <Flex
        as="button"
        flexDir="row"
        alignItems="center"
        gap={3}
        py={4}
        px={2}
        onClick={onToggleExpander}
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
