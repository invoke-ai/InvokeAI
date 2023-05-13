import { CheckIcon, ChevronUpIcon } from '@chakra-ui/icons';
import {
  Flex,
  FormControl,
  FormControlProps,
  FormLabel,
  Grid,
  GridItem,
  Input,
  List,
  ListItem,
  Select,
  Spacer,
  Text,
} from '@chakra-ui/react';
import { useEnsureOnScreen } from 'common/hooks/useEnsureOnScreen';
import { useSelect } from 'downshift';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';

import { memo, useRef } from 'react';
import { useIntersection } from 'react-use';

const BUTTON_BG = 'base.900';
const BORDER_HOVER = 'base.700';
const BORDER_FOCUS = 'accent.600';

type IAICustomSelectProps = {
  label?: string;
  items: string[];
  selectedItem: string;
  setSelectedItem: (v: string | null | undefined) => void;
  withCheckIcon?: boolean;
  formControlProps?: FormControlProps;
};

const IAICustomSelect = (props: IAICustomSelectProps) => {
  const {
    label,
    items,
    setSelectedItem,
    selectedItem,
    withCheckIcon,
    formControlProps,
  } = props;

  const {
    isOpen,
    getToggleButtonProps,
    getLabelProps,
    getMenuProps,
    highlightedIndex,
    getItemProps,
  } = useSelect({
    items,
    selectedItem,
    onSelectedItemChange: ({ selectedItem: newSelectedItem }) =>
      setSelectedItem(newSelectedItem),
  });

  const toggleButtonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLUListElement>(null);

  return (
    <FormControl {...formControlProps}>
      {label && (
        <FormLabel
          {...getLabelProps()}
          onClick={() => {
            toggleButtonRef.current && toggleButtonRef.current.focus();
          }}
        >
          {label}
        </FormLabel>
      )}
      <Select
        as={Flex}
        {...getToggleButtonProps({
          ref: toggleButtonRef,
        })}
        ref={toggleButtonRef}
        sx={{
          alignItems: 'center',
          userSelect: 'none',
          cursor: 'pointer',
        }}
      >
        <Text sx={{ fontSize: 'sm', fontWeight: 500, color: 'base.100' }}>
          {selectedItem}
        </Text>
      </Select>
      <List
        {...getMenuProps({ ref: menuRef })}
        as={Flex}
        sx={{
          position: 'absolute',
          visibility: isOpen ? 'visible' : 'hidden',
          flexDirection: 'column',
          zIndex: 1,
          bg: 'base.800',
          borderRadius: 'base',
          border: '1px',
          borderColor: 'base.700',
          shadow: 'dark-lg',
          py: 2,
          px: 0,
          h: 'fit-content',
          maxH: 64,
          mt: 1,
        }}
      >
        <OverlayScrollbarsComponent defer>
          {isOpen &&
            items.map((item, index) => (
              <ListItem
                sx={{
                  bg: highlightedIndex === index ? 'base.700' : undefined,
                  py: 1,
                  paddingInlineStart: 3,
                  paddingInlineEnd: 6,
                  cursor: 'pointer',
                  transitionProperty: 'common',
                  transitionDuration: '0.15s',
                }}
                key={`${item}${index}`}
                {...getItemProps({ item, index })}
              >
                {withCheckIcon ? (
                  <Grid gridTemplateColumns="1.25rem auto">
                    <GridItem>
                      {selectedItem === item && <CheckIcon boxSize={2} />}
                    </GridItem>
                    <GridItem>
                      <Text
                        sx={{
                          fontSize: 'sm',
                          color: 'base.100',
                          fontWeight: 500,
                        }}
                      >
                        {item}
                      </Text>
                    </GridItem>
                  </Grid>
                ) : (
                  <Text
                    sx={{
                      fontSize: 'sm',
                      color: 'base.100',
                      fontWeight: 500,
                    }}
                  >
                    {item}
                  </Text>
                )}
              </ListItem>
            ))}
        </OverlayScrollbarsComponent>
      </List>
    </FormControl>
  );
};

export default memo(IAICustomSelect);
