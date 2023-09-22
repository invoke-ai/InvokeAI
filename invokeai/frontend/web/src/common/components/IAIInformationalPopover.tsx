import {
  Box,
  Button,
  Divider,
  Flex,
  Heading,
  Image,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverCloseButton,
  PopoverContent,
  PopoverProps,
  PopoverTrigger,
  Portal,
  Text,
} from '@chakra-ui/react';
import { ReactNode, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useAppSelector } from '../../app/store/storeHooks';

const OPEN_DELAY = 1500;

type Props = Omit<PopoverProps, 'children'> & {
  details: string;
  children: ReactNode;
  image?: string;
  buttonLabel?: string;
  buttonHref?: string;
  placement?: PopoverProps['placement'];
};

const IAIInformationalPopover = ({
  details,
  image,
  buttonLabel,
  buttonHref,
  children,
  placement,
}: Props) => {
  const shouldEnableInformationalPopovers = useAppSelector(
    (state) => state.system.shouldEnableInformationalPopovers
  );
  const { t } = useTranslation();

  const heading = t(`popovers.${details}.heading`);
  const paragraph = t(`popovers.${details}.paragraph`);

  if (!shouldEnableInformationalPopovers) {
    return <>{children}</>;
  }

  return (
    <Popover
      placement={placement || 'top'}
      closeOnBlur={false}
      trigger="hover"
      variant="informational"
      openDelay={OPEN_DELAY}
    >
      <PopoverTrigger>
        <Box w="full">{children}</Box>
      </PopoverTrigger>
      <Portal>
        <PopoverContent>
          <PopoverArrow />
          <PopoverCloseButton />

          <PopoverBody>
            <Flex
              sx={{
                gap: 3,
                flexDirection: 'column',
                width: '100%',
                alignItems: 'center',
              }}
            >
              {image && (
                <Image
                  sx={{
                    objectFit: 'contain',
                    maxW: '60%',
                    maxH: '60%',
                    backgroundColor: 'white',
                  }}
                  src={image}
                  alt="Optional Image"
                />
              )}
              <Flex
                sx={{
                  gap: 3,
                  flexDirection: 'column',
                  width: '100%',
                }}
              >
                {heading && (
                  <>
                    <Heading size="sm">{heading}</Heading>
                    <Divider />
                  </>
                )}
                <Text>{paragraph}</Text>
                {buttonLabel && (
                  <Flex justifyContent="flex-end">
                    <Button
                      onClick={() => window.open(buttonHref)}
                      size="sm"
                      variant="invokeAIOutline"
                    >
                      {buttonLabel}
                    </Button>
                  </Flex>
                )}
              </Flex>
            </Flex>
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
};

export default memo(IAIInformationalPopover);
