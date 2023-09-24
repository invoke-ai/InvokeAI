import {
  Box,
  BoxProps,
  Button,
  Divider,
  Flex,
  Heading,
  Image,
  Popover,
  PopoverBody,
  PopoverCloseButton,
  PopoverContent,
  PopoverProps,
  PopoverTrigger,
  Portal,
  Text,
  forwardRef,
} from '@chakra-ui/react';
import { merge, omit } from 'lodash-es';
import { PropsWithChildren, memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaExternalLinkAlt } from 'react-icons/fa';
import { useAppSelector } from '../../../app/store/storeHooks';
import {
  Feature,
  OPEN_DELAY,
  POPOVER_DATA,
  POPPER_MODIFIERS,
} from './constants';

type Props = PropsWithChildren & {
  feature: Feature;
  wrapperProps?: BoxProps;
  popoverProps?: PopoverProps;
};

const IAIInformationalPopover = forwardRef(
  ({ feature, children, wrapperProps, ...rest }: Props, ref) => {
    const { t } = useTranslation();
    const shouldEnableInformationalPopovers = useAppSelector(
      (state) => state.system.shouldEnableInformationalPopovers
    );

    const data = useMemo(() => POPOVER_DATA[feature], [feature]);

    const popoverProps = useMemo(
      () => merge(omit(data, ['image', 'href', 'buttonLabel']), rest),
      [data, rest]
    );

    const heading = useMemo<string | undefined>(
      () => t(`popovers.${feature}.heading`),
      [feature, t]
    );

    const paragraphs = useMemo<string[]>(
      () =>
        t(`popovers.${feature}.paragraphs`, {
          returnObjects: true,
        }) ?? [],
      [feature, t]
    );

    const handleClick = useCallback(() => {
      if (!data?.href) {
        return;
      }
      window.open(data.href);
    }, [data?.href]);

    if (!shouldEnableInformationalPopovers) {
      return (
        <Box ref={ref} w="full" {...wrapperProps}>
          {children}
        </Box>
      );
    }

    return (
      <Popover
        isLazy
        closeOnBlur={false}
        trigger="hover"
        variant="informational"
        openDelay={OPEN_DELAY}
        modifiers={POPPER_MODIFIERS}
        placement="top"
        {...popoverProps}
      >
        <PopoverTrigger>
          <Box ref={ref} w="full" {...wrapperProps}>
            {children}
          </Box>
        </PopoverTrigger>
        <Portal>
          <PopoverContent w={96}>
            <PopoverCloseButton />
            <PopoverBody>
              <Flex
                sx={{
                  gap: 2,
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                }}
              >
                {heading && (
                  <>
                    <Heading size="sm">{heading}</Heading>
                    <Divider />
                  </>
                )}
                {data?.image && (
                  <>
                    <Image
                      sx={{
                        objectFit: 'contain',
                        maxW: '60%',
                        maxH: '60%',
                        backgroundColor: 'white',
                      }}
                      src={data.image}
                      alt="Optional Image"
                    />
                    <Divider />
                  </>
                )}
                {paragraphs.map((p) => (
                  <Text key={p}>{p}</Text>
                ))}
                {data?.href && (
                  <>
                    <Divider />
                    <Button
                      pt={1}
                      onClick={handleClick}
                      leftIcon={<FaExternalLinkAlt />}
                      alignSelf="flex-end"
                      variant="link"
                    >
                      {t('common.learnMore') ?? heading}
                    </Button>
                  </>
                )}
              </Flex>
            </PopoverBody>
          </PopoverContent>
        </Portal>
      </Popover>
    );
  }
);

IAIInformationalPopover.displayName = 'IAIInformationalPopover';

export default memo(IAIInformationalPopover);
