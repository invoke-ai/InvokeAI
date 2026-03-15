import { Flex, Image, Spinner } from "@invoke-ai/ui-library";
import { memo } from "react";
import { imagesApi } from "services/api/endpoints/images";

import type { StageFeedBoardItem, StageFeedQueueItem } from "./common";

export const StageViewerGridBoardItem = memo(({ item }: { item: StageFeedBoardItem }) => {
  const imageName = item.id;
  const { currentData: imageDTO, isUninitialized } = imagesApi.endpoints.getImageDTO.useQueryState(imageName);
  imagesApi.endpoints.getImageDTO.useQuerySubscription(imageName, { skip: isUninitialized });

  const aspectRatio = imageDTO ? imageDTO.width / imageDTO.height : 1;

  return (
    <Flex
      flexDir="column"
    >
      <Flex
        width="100%"
        aspectRatio={aspectRatio}
        bg="gray.700"
        alignItems="center"
        justifyContent="center"
      >
        {imageDTO && (
          <Image
            src={imageDTO.image_url}
            alt={imageDTO.image_name}
            objectFit="cover"
            width="100%"
            height="100%"
          />
        )}
      </Flex>

      <Flex>
        {imageDTO ? imageDTO.image_name : 'Loading...'}
      </Flex>
    </Flex>
  );
});

StageViewerGridBoardItem.displayName = 'StageViewerGridBoardItem';

export const StageViewerGridQueueItem = memo(({ item }: {item: StageFeedQueueItem}) => {
  const queueItemId = item.id;

  return (
    <Flex flexDir="column">
      <Flex width="100%" aspectRatio={1} bg="gray.700" alignItems="center" justifyContent="center">
        <Spinner size="lg" />
      </Flex>
      <Flex>
        Queue Item: {queueItemId}
      </Flex>
    </Flex>
  );
});

StageViewerGridQueueItem.displayName = 'StageViewerGridQueueItem';
