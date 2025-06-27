import { IconButton, Popover, PopoverBody, PopoverContent, PopoverTrigger, Text, VStack } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';

export const StagingAreaToolbarInfoButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const ctx = useCanvasSessionContext();
  const selectedItem = useStore(ctx.$selectedItem);
  const { t } = useTranslation();

  const formatTimestamp = useCallback((timestamp: string | null | undefined) => {
    if (!timestamp) {
      return 'N/A';
    }
    return new Date(timestamp).toLocaleString();
  }, []);

  const formatDuration = useCallback((start: string | null | undefined, end: string | null | undefined) => {
    if (!start || !end) {
      return 'N/A';
    }
    const duration = new Date(end).getTime() - new Date(start).getTime();
    return `${(duration / 1000).toFixed(2)}s`;
  }, []);

  if (!selectedItem) {
    return (
      <IconButton
        tooltip={t('controlLayers.stagingArea.info')}
        aria-label={t('controlLayers.stagingArea.info')}
        icon={<PiInfoBold />}
        colorScheme="invokeBlue"
        isDisabled={true}
      />
    );
  }

  return (
    <Popover placement="top" isLazy>
      <PopoverTrigger>
        <IconButton
          tooltip={t('controlLayers.stagingArea.info')}
          aria-label={t('controlLayers.stagingArea.info')}
          icon={<PiInfoBold />}
          colorScheme="invokeBlue"
          isDisabled={isDisabled}
        />
      </PopoverTrigger>
      <PopoverContent maxW="300px">
        <PopoverBody>
          <VStack align="start" spacing={2} fontSize="sm">
            <Text fontWeight="bold">Generation Info</Text>

            <VStack align="start" spacing={1} w="full">
              <Text>
                <Text as="span" fontWeight="semibold">
                  Status:
                </Text>{' '}
                {selectedItem.status}
              </Text>
              <Text>
                <Text as="span" fontWeight="semibold">
                  Item ID:
                </Text>{' '}
                {selectedItem.item_id}
              </Text>
              <Text>
                <Text as="span" fontWeight="semibold">
                  Priority:
                </Text>{' '}
                {selectedItem.priority}
              </Text>

              {selectedItem.origin && (
                <Text>
                  <Text as="span" fontWeight="semibold">
                    Origin:
                  </Text>{' '}
                  {selectedItem.origin}
                </Text>
              )}

              {selectedItem.destination && (
                <Text>
                  <Text as="span" fontWeight="semibold">
                    Destination:
                  </Text>{' '}
                  {selectedItem.destination}
                </Text>
              )}

              <Text>
                <Text as="span" fontWeight="semibold">
                  Created:
                </Text>{' '}
                {formatTimestamp(selectedItem.created_at)}
              </Text>

              {selectedItem.started_at && (
                <Text>
                  <Text as="span" fontWeight="semibold">
                    Started:
                  </Text>{' '}
                  {formatTimestamp(selectedItem.started_at)}
                </Text>
              )}

              {selectedItem.completed_at && (
                <Text>
                  <Text as="span" fontWeight="semibold">
                    Completed:
                  </Text>{' '}
                  {formatTimestamp(selectedItem.completed_at)}
                </Text>
              )}

              {selectedItem.started_at && selectedItem.completed_at && (
                <Text>
                  <Text as="span" fontWeight="semibold">
                    Duration:
                  </Text>{' '}
                  {formatDuration(selectedItem.started_at, selectedItem.completed_at)}
                </Text>
              )}

              {selectedItem.credits && (
                <Text>
                  <Text as="span" fontWeight="semibold">
                    Credits:
                  </Text>{' '}
                  {selectedItem.credits}
                </Text>
              )}

              {selectedItem.error_message && (
                <Text color="error.300">
                  <Text as="span" fontWeight="semibold">
                    Error:
                  </Text>{' '}
                  {selectedItem.error_message}
                </Text>
              )}
            </VStack>
          </VStack>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

StagingAreaToolbarInfoButton.displayName = 'StagingAreaToolbarInfoButton';
