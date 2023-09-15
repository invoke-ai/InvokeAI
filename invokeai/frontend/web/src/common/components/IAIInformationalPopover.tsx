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
import { ReactNode } from 'react';
import { useAppSelector } from '../../app/store/storeHooks';
import { systemSelector } from '../../features/system/store/systemSelectors';

interface Props extends PopoverProps {
  heading?: string;
  paragraph: string;
  triggerComponent: ReactNode;
  image?: string;
  buttonLabel?: string;
  buttonHref?: string;
  placement?: string;
}

function IAIInformationalPopover({
  heading,
  paragraph,
  image,
  buttonLabel,
  buttonHref,
  triggerComponent,
  placement,
}: Props) {
  const { shouldDisableInformationalPopovers } = useAppSelector(systemSelector);

  if (shouldDisableInformationalPopovers) {
    return triggerComponent;
  } else {
    return (
      <Popover
        placement={placement || 'top'}
        closeOnBlur={false}
        trigger="hover"
        variant="informational"
      >
        <PopoverTrigger>
          <div>{triggerComponent}</div>
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
