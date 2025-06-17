import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Divider,
  Flex,
  IconButton,
  Image,
  Popover,
  PopoverAnchor,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  Portal,
  Skeleton,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { POPPER_MODIFIERS } from 'common/components/InformationalPopover/constants';
import type { UseDisclosure } from 'common/hooks/useBoolean';
import { useDisclosure } from 'common/hooks/useBoolean';
import { DEFAULT_FILTER, useFilterableOutsideClick } from 'common/hooks/useFilterableOutsideClick';
import { RefImageHeader } from 'features/controlLayers/components/RefImage/RefImageHeader';
import { RefImageSettings } from 'features/controlLayers/components/RefImage/RefImageSettings';
import { useRefImageEntity } from 'features/controlLayers/components/RefImage/useRefImageEntity';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { memo, useCallback, useRef } from 'react';
import { PiImageBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

// There is some awkwardness here with closing the popover when clicking outside of it, related to Chakra's
// handling of refs, portals, outside clicks, and a race condition with framer-motion animations that can leave
// the popover closed when its internal state is still open.
//
// We have to manually manage the popover open state to work around the race condition, and then have to do special
// handling to close the popover when clicking outside of it.

// We have to reach outside react to identify the popover trigger element instead of using refs, thanks to how Chakra
// handles refs for PopoverAnchor internally. Maybe there is some way to merge them but I couldn't figure it out.
const getRefImagePopoverTriggerId = (id: string) => `ref-image-popover-trigger-${id}`;

export const RefImage = memo(() => {
  const id = useRefImageIdContext();
  const ref = useRef<HTMLDivElement>(null);
  const disclosure = useDisclosure(false);
  // This filter prevents the popover from closing when clicking on a sibling portal element, like the dropdown menu
  // inside the ref image settings popover. It also prevents the popover from closing when clicking on the popover's
  // own trigger element.
  const filter = useCallback(
    (el: HTMLElement | SVGElement) => {
      return DEFAULT_FILTER(el) || el.id === getRefImagePopoverTriggerId(id);
    },
    [id]
  );
  useFilterableOutsideClick({ ref, handler: disclosure.close, filter });

  return (
    <Popover
      // The popover contains a react-select component, which uses a portal to render its options. This portal
      // is itself not lazy. As a result, if we do not unmount the popover when it is closed, the react-select
      // component still exists but is invisible, and intercepts clicks!
      isLazy
      lazyBehavior="unmount"
      isOpen={disclosure.isOpen}
      closeOnBlur={false}
      modifiers={POPPER_MODIFIERS}
    >
      <Thumbnail disclosure={disclosure} />
      <Portal>
        <PopoverContent ref={ref} w={400}>
          <PopoverArrow />
          <PopoverBody>
            <Flex flexDir="column" gap={2} w="full" h="full">
              <RefImageHeader />
              <Divider />
              <RefImageSettings />
            </Flex>
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});
RefImage.displayName = 'RefImage';

const imageSx: SystemStyleObject = {
  opacity: 0.7,
  transitionProperty: 'opacity',
  transitionDuration: 'normal',
  _hover: {
    opacity: 1,
  },
  '&[data-is-open="true"]': {
    opacity: 1,
  },
};

const Thumbnail = memo(({ disclosure }: { disclosure: UseDisclosure }) => {
  const id = useRefImageIdContext();
  const entity = useRefImageEntity(id);
  const { data: imageDTO } = useGetImageDTOQuery(entity.config.image?.image_name ?? skipToken);

  if (!entity.config.image) {
    return (
      <PopoverAnchor>
        <IconButton
          id={getRefImagePopoverTriggerId(id)}
          aria-label="Open Reference Image Settings"
          h="full"
          variant="ghost"
          aspectRatio="1/1"
          borderWidth="2px !important"
          borderStyle="dashed !important"
          borderColor="errorAlpha.500"
          borderRadius="base"
          icon={<PiImageBold />}
          colorScheme="error"
          onClick={disclosure.toggle}
          flexShrink={0}
        />
      </PopoverAnchor>
    );
  }
  return (
    <PopoverAnchor>
      <Image
        borderWidth={1}
        borderStyle="solid"
        id={getRefImagePopoverTriggerId(id)}
        role="button"
        src={imageDTO?.thumbnail_url}
        objectFit="contain"
        aspectRatio="1/1"
        // width={imageDTO?.width}
        height={imageDTO?.height}
        fallback={<Skeleton h="full" aspectRatio="1/1" />}
        maxW="full"
        maxH="full"
        borderRadius="base"
        onClick={disclosure.toggle}
        flexShrink={0}
        sx={imageSx}
        data-is-open={disclosure.isOpen}
      />
    </PopoverAnchor>
  );
});
Thumbnail.displayName = 'Thumbnail';
