import {
  Button,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverArrow,
  PopoverCloseButton,
  PopoverHeader,
  PopoverBody,
  PopoverProps,
  Flex,
  Text,
  Image,
} from '@chakra-ui/react';
import { useAppSelector } from '../../app/store/storeHooks';
import { systemSelector } from '../../features/system/store/systemSelectors';
import { useTranslation } from 'react-i18next';

interface Props extends PopoverProps {
  details: string;
  children: JSX.Element;
  image?: string;
  buttonLabel?: string;
  buttonHref?: string;
  placement?: PopoverProps['placement'];
}

function IAIInformationalPopover({
  details,
  image,
  buttonLabel,
  buttonHref,
  children,
  placement,
}: Props): JSX.Element {
  const { shouldDisableInformationalPopovers } = useAppSelector(systemSelector);
  const { t } = useTranslation();

  const heading = t(`popovers.${details}.heading`);
  const paragraph = t(`popovers.${details}.paragraph`);

  if (shouldDisableInformationalPopovers) {
    return children;
  } else {
    return (
      <Popover
        placement={placement || 'top'}
        closeOnBlur={false}
        trigger="hover"
        variant="informational"
      >
        <PopoverTrigger>
          <div>{children}</div>
        </PopoverTrigger>
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
                  p: 3,
                  pt: heading ? 0 : 3,
                }}
              >
                {heading && <PopoverHeader>{heading}</PopoverHeader>}
                <Text sx={{ px: 3 }}>{paragraph}</Text>
                {buttonLabel && (
                  <Flex sx={{ px: 3 }} justifyContent="flex-end">
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
      </Popover>
    );
  }
}

export default IAIInformationalPopover;
