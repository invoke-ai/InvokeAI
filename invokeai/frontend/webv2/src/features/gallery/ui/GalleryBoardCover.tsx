import type { GalleryBoard } from '@features/gallery/core/types';

import { Flex, Icon, Image } from '@chakra-ui/react';
import { CalendarIcon, ImageIcon, PlayIcon, type LucideIcon } from 'lucide-react';

/** Square swatch shown at the start of every board row and in the stacked header. */
export const BoardCoverIcon = ({ icon }: { icon: LucideIcon }) => (
  <Flex
    align="center"
    bg="bg.emphasized"
    borderColor="border.subtle"
    borderWidth="1px"
    boxSize="5"
    color="fg.subtle"
    flexShrink={0}
    justify="center"
    rounded="sm"
  >
    <Icon as={icon} boxSize="3" />
  </Flex>
);

export const BoardCover = ({ board }: { board: GalleryBoard }) => {
  if (board.coverThumbnailUrl) {
    return (
      <Flex boxSize="5" flexShrink={0} position="relative">
        <Image
          alt=""
          bg="bg.emphasized"
          borderColor="border.subtle"
          borderWidth="1px"
          boxSize="5"
          objectFit="cover"
          rounded="sm"
          src={board.coverThumbnailUrl}
        />
        {board.coverVideoName ? (
          <Flex
            align="center"
            aria-hidden="true"
            bg="blackAlpha.700"
            bottom="0"
            boxSize="2.5"
            color="white"
            justify="center"
            position="absolute"
            right="0"
            rounded="full"
          >
            <Icon as={PlayIcon} boxSize="1.5" fill="currentColor" />
          </Flex>
        ) : null}
      </Flex>
    );
  }

  return <BoardCoverIcon icon={board.kind === 'date' ? CalendarIcon : ImageIcon} />;
};
