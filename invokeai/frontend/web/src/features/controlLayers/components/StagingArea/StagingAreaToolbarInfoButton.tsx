import {
  Divider,
  Grid,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Text,
  VStack,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { MetadataItem } from 'features/metadata/components/MetadataItem';
import { MetadataLoRAs } from 'features/metadata/components/MetadataLoRAs';
import { useMetadataExtraction } from 'features/metadata/hooks/useMetadataExtraction';
import { handlers } from 'features/metadata/util/handlers';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';

export const StagingAreaToolbarInfoButton = memo(({ isDisabled }: { isDisabled?: boolean }) => {
  const ctx = useCanvasSessionContext();
  const selectedItem = useStore(ctx.$selectedItem);
  const { t } = useTranslation();

  // Extract metadata using the unified hook
  const metadata = useMetadataExtraction(selectedItem);

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
      <PopoverContent maxW="500px" bg="base.900" borderColor="base.700">
        <PopoverBody p={4}>
          <VStack align="start" spacing={4} fontSize="sm">
            {/* Prompts Section */}
            <VStack align="start" spacing={3} w="full">
              <Text fontWeight="semibold" fontSize="md" color="base.100">
                Prompts
              </Text>

              {metadata !== null && (
                <>
                  <MetadataItem
                    metadata={metadata}
                    handlers={handlers.positivePrompt}
                    displayMode="card"
                    showCopy={true}
                  />
                  <MetadataItem
                    metadata={metadata}
                    handlers={handlers.negativePrompt}
                    displayMode="card"
                    showCopy={true}
                  />
                </>
              )}
            </VStack>

            <Divider borderColor="base.700" />

            {/* Models and LoRAs Section - Left Column */}
            <Grid templateColumns="1fr 1fr" gap={6} w="full">
              <VStack align="start" spacing={4} w="full">
                {/* Model Section */}
                <VStack align="start" spacing={3} w="full">
                  <Text fontWeight="semibold" fontSize="md" color="base.100">
                    Model
                  </Text>

                  {metadata !== null && (
                    <VStack align="start" spacing={2} w="full">
                      <MetadataItem
                        metadata={metadata}
                        handlers={handlers.model}
                        displayMode="badge"
                        colorScheme="invokeBlue"
                        showCopy={true}
                      />
                      <MetadataItem
                        metadata={metadata}
                        handlers={handlers.vae}
                        displayMode="badge"
                        colorScheme="base"
                        showCopy={true}
                      />
                    </VStack>
                  )}
                </VStack>

                {/* LoRA Section */}
                {metadata !== null && <MetadataLoRAs metadata={metadata} displayMode="badge" showCopy={true} />}
              </VStack>

              {/* Other Settings Section - Right Column */}
              <VStack align="start" spacing={3} w="full">
                <Text fontWeight="semibold" fontSize="md" color="base.100">
                  Other Settings
                </Text>

                {metadata !== null && (
                  <VStack align="start" spacing={3} w="full">
                    <MetadataItem metadata={metadata} handlers={handlers.seed} displayMode="simple" showCopy={true} />
                    <MetadataItem metadata={metadata} handlers={handlers.steps} displayMode="simple" showCopy={true} />
                    <MetadataItem
                      metadata={metadata}
                      handlers={handlers.cfgScale}
                      displayMode="simple"
                      showCopy={true}
                    />
                    <MetadataItem
                      metadata={metadata}
                      handlers={handlers.scheduler}
                      displayMode="simple"
                      showCopy={true}
                    />
                  </VStack>
                )}
              </VStack>
            </Grid>

            {/* Error Section */}
            {selectedItem.error_message && (
              <>
                <Divider borderColor="base.700" />
                <VStack align="start" spacing={2} w="full">
                  <Text fontWeight="semibold" fontSize="md" color="error.300">
                    Error
                  </Text>
                  <Text
                    fontSize="sm"
                    color="error.200"
                    bg="error.900"
                    p={3}
                    borderRadius="lg"
                    w="full"
                    border="1px solid"
                    borderColor="error.700"
                  >
                    {selectedItem.error_message}
                  </Text>
                </VStack>
              </>
            )}
          </VStack>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

StagingAreaToolbarInfoButton.displayName = 'StagingAreaToolbarInfoButton';
