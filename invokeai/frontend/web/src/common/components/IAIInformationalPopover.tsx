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

interface Props extends PopoverProps {
  heading: string;
  paragraph: string;
  triggerComponent: ReactNode;
  image?: string;
  buttonLabel?: string;
  buttonHref?: string;
}

function IAIInformationalPopover({
  heading,
  paragraph,
  image,
  buttonLabel,
  buttonHref,
  triggerComponent,
}: Props) {
  return (
    <Popover
      placement="top"
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
            }}
          >
            {image && (
              <Image
                sx={{
                  objectFit: 'contain',
                  maxW: '100%',
                  maxH: 'full',
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
                pt: 0,
              }}
            >
              <PopoverHeader>{heading}</PopoverHeader>
              <Text sx={{ pl: 3, pr: 3 }}>{paragraph}</Text>
              {buttonLabel && (
                <Flex sx={{ pl: 3, pr: 3 }} justifyContent="flex-end">
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

export default IAIInformationalPopover;
