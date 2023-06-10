import { CheckIcon, ChevronUpIcon } from '@chakra-ui/icons';
import {
  Box,
  Flex,
  FormControl,
  FormControlProps,
  FormLabel,
  Grid,
  GridItem,
  List,
  ListItem,
  Text,
  Tooltip,
  TooltipProps,
} from '@chakra-ui/react';
import { autoUpdate, offset, shift, useFloating } from '@floating-ui/react-dom';
import { useSelect } from 'downshift';
import { isString } from 'lodash-es';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';

import { memo, useMemo } from 'react';
import { getInputOutlineStyles } from 'theme/util/getInputOutlineStyles';

export type ItemTooltips = { [key: string]: string };

export type IAICustomSelectOption = {
  value: string;
  label: string;
  tooltip?: string;
};

type IAICustomSelectProps = {
  label?: string;
  value: string;
  data: IAICustomSelectOption[] | string[];
  onChange: (v: string) => void;
  withCheckIcon?: boolean;
  formControlProps?: FormControlProps;
  tooltip?: string;
  tooltipProps?: Omit<TooltipProps, 'children'>;
  ellipsisPosition?: 'start' | 'end';
};

const IAICustomSelect = (props: IAICustomSelectProps) => {
  const {
    label,
    withCheckIcon,
    formControlProps,
    tooltip,
    tooltipProps,
    ellipsisPosition = 'end',
    data,
    value,
    onChange,
  } = props;

  const values = useMemo(() => {
    return data.map<IAICustomSelectOption>((v) => {
      if (isString(v)) {
        return { value: v, label: v };
      }
      return v;
    });
  }, [data]);

  const stringValues = useMemo(() => {
    return values.map((v) => v.value);
  }, [values]);

  const valueData = useMemo(() => {
    return values.find((v) => v.value === value);
  }, [values, value]);

  const {
    isOpen,
    getToggleButtonProps,
    getLabelProps,
    getMenuProps,
    highlightedIndex,
    getItemProps,
  } = useSelect({
    items: stringValues,
    selectedItem: value,
    onSelectedItemChange: ({ selectedItem: newSelectedItem }) => {
      newSelectedItem && onChange(newSelectedItem);
    },
  });

  const { refs, floatingStyles } = useFloating<HTMLButtonElement>({
    whileElementsMounted: autoUpdate,
    middleware: [offset(4), shift({ crossAxis: true, padding: 8 })],
  });

  const labelTextDirection = useMemo(() => {
    if (ellipsisPosition === 'start') {
      return document.dir === 'rtl' ? 'ltr' : 'rtl';
    }

    return document.dir;
  }, [ellipsisPosition]);

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
        <Flex
          {...getToggleButtonProps({ ref: refs.reference })}
          sx={{
            alignItems: 'center',
            userSelect: 'none',
            cursor: 'pointer',
            overflow: 'hidden',
            width: 'full',
            py: 1,
            px: 2,
            gap: 2,
            justifyContent: 'space-between',
            ...getInputOutlineStyles(),
          }}
        >
          <Text
            sx={{
              fontSize: 'sm',
              fontWeight: 500,
              color: 'base.100',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              direction: labelTextDirection,
            }}
          >
            {valueData?.label}
          </Text>
          <ChevronUpIcon
            sx={{
              color: 'base.300',
              transform: isOpen ? 'rotate(0deg)' : 'rotate(180deg)',
              transitionProperty: 'common',
              transitionDuration: 'normal',
            }}
          />
        </Flex>
      </Tooltip>
      <Box {...getMenuProps()}>
        {isOpen && (
          <List
            as={Flex}
            ref={refs.floating}
            sx={{
              ...floatingStyles,
              top: 0,
              insetInlineStart: 0,
              flexDirection: 'column',
              zIndex: 2,
              bg: 'base.800',
              borderRadius: 'base',
              border: '1px',
              borderColor: 'base.700',
              shadow: 'dark-lg',
              py: 2,
              px: 0,
              h: 'fit-content',
              maxH: 64,
              minW: 48,
            }}
          >
            <OverlayScrollbarsComponent>
              {values.map((v, index) => {
                const isSelected = value === v.value;
                const isHighlighted = highlightedIndex === index;
                const fontWeight = isSelected ? 700 : 500;
                const bg = isHighlighted
                  ? 'base.700'
                  : isSelected
                  ? 'base.750'
                  : undefined;
                return (
                  <Tooltip
                    isDisabled={!v.tooltip}
                    key={`${v.value}${index}`}
                    label={v.tooltip}
                    hasArrow
                    placement="right"
                  >
                    <ListItem
                      sx={{
                        bg,
                        py: 1,
                        paddingInlineStart: 3,
                        paddingInlineEnd: 6,
                        cursor: 'pointer',
                        transitionProperty: 'common',
                        transitionDuration: '0.15s',
                      }}
                      {...getItemProps({ item: v.value, index })}
                    >
                      {withCheckIcon ? (
                        <Grid gridTemplateColumns="1.25rem auto">
                          <GridItem>
                            {isSelected && <CheckIcon boxSize={2} />}
                          </GridItem>
                          <GridItem>
                            <Text
                              sx={{
                                fontSize: 'sm',
                                color: 'base.100',
                                fontWeight,
                              }}
                            >
                              {v.label}
                            </Text>
                          </GridItem>
                        </Grid>
                      ) : (
                        <Text
                          sx={{
                            fontSize: 'sm',
                            color: 'base.50',
                            fontWeight,
                          }}
                        >
                          {v.label}
                        </Text>
                      )}
                    </ListItem>
                  </Tooltip>
                );
              })}
            </OverlayScrollbarsComponent>
          </List>
        )}
      </Box>
    </FormControl>
  );
};

export default memo(IAICustomSelect);
