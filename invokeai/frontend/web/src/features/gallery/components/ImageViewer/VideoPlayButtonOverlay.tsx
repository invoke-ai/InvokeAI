import { Flex, Icon } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { PiPlayFill } from 'react-icons/pi';

type Props = {
  onClick: () => void;
};

/**
 * Large centered play button shown over the still thumbnail in the video viewer. Clicking it
 * swaps the preview into HTML5 video playback (see CurrentVideoPreview).
 */
export const VideoPlayButtonOverlay = memo(({ onClick }: Props) => {
  return (
    <Flex
      position="absolute"
      top="50%"
      left="50%"
      transform="translate(-50%, -50%)"
      as="button"
      onClick={onClick}
      aria-label="Play video"
      alignItems="center"
      justifyContent="center"
      bg="blackAlpha.500"
      _hover={{ bg: 'blackAlpha.700' }}
      transition="background 120ms ease"
      borderRadius="full"
      boxSize="96px"
      cursor="pointer"
      boxShadow="0 4px 12px rgba(0,0,0,0.4)"
    >
      <Icon as={PiPlayFill} boxSize="56px" color="white" filter="drop-shadow(0 0 4px rgba(0,0,0,0.6))" />
    </Flex>
  );
});

VideoPlayButtonOverlay.displayName = 'VideoPlayButtonOverlay';
