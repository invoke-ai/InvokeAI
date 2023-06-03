import { CheckIcon } from '@chakra-ui/icons';
import {
  Box,
  Flex,
  FlexProps,
  FormControl,
  FormControlProps,
  FormLabel,
  Grid,
  GridItem,
  List,
  ListItem,
  Select,
  Text,
  Tooltip,
  TooltipProps,
} from '@chakra-ui/react';
import { autoUpdate, offset, shift, useFloating } from '@floating-ui/react-dom';
import { useSelect } from 'downshift';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';

import { memo } from 'react';

export type ItemTooltips = { [key: string]: string };

type IAICustomSelectProps = {
  label?: string;
  items: string[];
  itemTooltips?: ItemTooltips;
  selectedItem: string;
  setSelectedItem: (v: string | null | undefined) => void;
  withCheckIcon?: boolean;
  formControlProps?: FormControlProps;
  buttonProps?: FlexProps;
  tooltip?: string;
  tooltipProps?: Omit<TooltipProps, 'children'>;
};

const IAICustomSelect = (props: IAICustomSelectProps) => {
  const {
    label,
    items,
    itemTooltips,
    setSelectedItem,
    selectedItem,
    withCheckIcon,
    formControlProps,
    tooltip,
    buttonProps,
    tooltipProps,
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

  const { refs, floatingStyles } = useFloating<HTMLButtonElement>({
    whileElementsMounted: autoUpdate,
    middleware: [offset(4), shift({ crossAxis: true, padding: 8 })],
  });

  return (
    <FormControl sx={{ w: 'full' }} {...formControlProps}>
      {label && (
        <FormLabel
          {...getLabelProps()}
          onClick={() => {
            refs.floating.current && refs.floating.current.focus();
          }}
        >
          {label}
        </FormLabel>
      )}
      <Tooltip label={tooltip} {...tooltipProps}>
        <Select
          {...getToggleButtonProps({ ref: refs.setReference })}
          {...buttonProps}
          as={Flex}
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
      </Tooltip>
      <Box {...getMenuProps()}>
        {isOpen && (
          <List
            as={Flex}
            ref={refs.setFloating}
            sx={{
              ...floatingStyles,
              width: 'max-content',
              top: 0,
              left: 0,
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
            }}
          >
            <OverlayScrollbarsComponent>
              {items.map((item, index) => (
                <Tooltip
                  isDisabled={!itemTooltips}
                  key={`${item}${index}`}
                  label={itemTooltips?.[item]}
                  hasArrow
                  placement="right"
                >
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
                </Tooltip>
              ))}
            </OverlayScrollbarsComponent>
          </List>
        )}
      </Box>
    </FormControl>
  );
};

export default memo(IAICustomSelect);
