import { ExternalLinkIcon } from '@chakra-ui/icons';
import {
  Box,
  Center,
  Flex,
  Heading,
  IconButton,
  Link,
  Text,
  Tooltip,
} from '@chakra-ui/react';
import * as InvokeAI from 'app/invokeai';
import { useAppDispatch } from 'app/storeHooks';
import { useGetUrl } from 'common/util/getUrl';
import promptToString from 'common/util/promptToString';
import { seedWeightsToString } from 'common/util/seedWeightPairs';
import useSetBothPrompts from 'features/parameters/hooks/usePrompt';
import {
  setCfgScale,
  setHeight,
  setImg2imgStrength,
  // setInitialImage,
  setMaskPath,
  setPerlin,
  setSampler,
  setSeamless,
  setSeed,
  setSeedWeights,
  setShouldFitToWidthHeight,
  setSteps,
  setThreshold,
  setWidth,
} from 'features/parameters/store/generationSlice';
import {
  setCodeformerFidelity,
  setFacetoolStrength,
  setFacetoolType,
  setHiresFix,
  setUpscalingDenoising,
  setUpscalingLevel,
  setUpscalingStrength,
} from 'features/parameters/store/postprocessingSlice';
import { setShouldShowImageDetails } from 'features/ui/store/uiSlice';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCopy } from 'react-icons/fa';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';

type MetadataItemProps = {
  isLink?: boolean;
  label: string;
  onClick?: () => void;
  value: number | string | boolean;
  labelPosition?: string;
  withCopy?: boolean;
};

/**
 * Component to display an individual metadata item or parameter.
 */
const MetadataItem = ({
  label,
  value,
  onClick,
  isLink,
  labelPosition,
  withCopy = false,
}: MetadataItemProps) => {
  const { t } = useTranslation();

  return (
    <Flex gap={2}>
      {onClick && (
        <Tooltip label={`Recall ${label}`}>
          <IconButton
            aria-label={t('accessibility.useThisParameter')}
            icon={<IoArrowUndoCircleOutline />}
            size="xs"
            variant="ghost"
            fontSize={20}
            onClick={onClick}
          />
        </Tooltip>
      )}
      {withCopy && (
        <Tooltip label={`Copy ${label}`}>
          <IconButton
            aria-label={`Copy ${label}`}
            icon={<FaCopy />}
            size="xs"
            variant="ghost"
            fontSize={14}
            onClick={() => navigator.clipboard.writeText(value.toString())}
          />
        </Tooltip>
      )}
      <Flex direction={labelPosition ? 'column' : 'row'}>
        <Text fontWeight="semibold" whiteSpace="pre-wrap" pr={2}>
          {label}:
        </Text>
        {isLink ? (
          <Link href={value.toString()} isExternal wordBreak="break-all">
            {value.toString()} <ExternalLinkIcon mx="2px" />
          </Link>
        ) : (
          <Text overflowY="scroll" wordBreak="break-all">
            {value.toString()}
          </Text>
        )}
      </Flex>
    </Flex>
  );
};

type ImageMetadataViewerProps = {
  image: InvokeAI.Image;
};

// TODO: I don't know if this is needed.
const memoEqualityCheck = (
  prev: ImageMetadataViewerProps,
  next: ImageMetadataViewerProps
) => prev.image.name === next.image.name;

// TODO: Show more interesting information in this component.

/**
 * Image metadata viewer overlays currently selected image and provides
 * access to any of its metadata for use in processing.
 */
const ImageMetadataViewer = memo(({ image }: ImageMetadataViewerProps) => {
  const dispatch = useAppDispatch();

  const setBothPrompts = useSetBothPrompts();

  useHotkeys('esc', () => {
    dispatch(setShouldShowImageDetails(false));
  });

  const sessionId = image.metadata.invokeai?.session_id;
  const invocation = image.metadata.invokeai?.invocation;

  const { t } = useTranslation();
  const { getUrl } = useGetUrl();

  const metadataJSON = JSON.stringify(image, null, 2);

  return (
    <Flex
      sx={{
        padding: 4,
        gap: 1,
        flexDirection: 'column',
        width: 'full',
        height: 'full',
        backdropFilter: 'blur(20px)',
        bg: 'whiteAlpha.600',
        _dark: {
          bg: 'blackAlpha.600',
        },
      }}
    >
      <Flex gap={2}>
        <Text fontWeight="semibold">File:</Text>
        <Link href={getUrl(image.url)} isExternal maxW="calc(100% - 3rem)">
          {image.url.length > 64
            ? image.url.substring(0, 64).concat('...')
            : image.url}
          <ExternalLinkIcon mx="2px" />
        </Link>
      </Flex>
      {Object.keys(invocation).length > 0 ? (
        <>
          {invocation.type && (
            <MetadataItem label="Invocation type" value={invocation.type} />
          )}
          {invocation.model && (
            <MetadataItem label="Model" value={invocation.model} />
          )}
          {invocation.prompt && (
            <MetadataItem
              label="Prompt"
              labelPosition="top"
              value={
                typeof invocation.prompt === 'string'
                  ? invocation.prompt
                  : promptToString(invocation.prompt)
              }
              onClick={() => setBothPrompts(invocation.prompt)}
            />
          )}
          {invocation.seed !== undefined && (
            <MetadataItem
              label="Seed"
              value={invocation.seed}
              onClick={() => dispatch(setSeed(invocation.seed))}
            />
          )}
          {invocation.threshold !== undefined && (
            <MetadataItem
              label="Noise Threshold"
              value={invocation.threshold}
              onClick={() => dispatch(setThreshold(invocation.threshold))}
            />
          )}
          {invocation.perlin !== undefined && (
            <MetadataItem
              label="Perlin Noise"
              value={invocation.perlin}
              onClick={() => dispatch(setPerlin(invocation.perlin))}
            />
          )}
          {invocation.scheduler && (
            <MetadataItem
              label="Sampler"
              value={invocation.scheduler}
              onClick={() => dispatch(setSampler(invocation.scheduler))}
            />
          )}
          {invocation.steps && (
            <MetadataItem
              label="Steps"
              value={invocation.steps}
              onClick={() => dispatch(setSteps(invocation.steps))}
            />
          )}
          {invocation.cfg_scale !== undefined && (
            <MetadataItem
              label="CFG scale"
              value={invocation.cfg_scale}
              onClick={() => dispatch(setCfgScale(invocation.cfg_scale))}
            />
          )}
          {invocation.variations && invocation.variations.length > 0 && (
            <MetadataItem
              label="Seed-weight pairs"
              value={seedWeightsToString(invocation.variations)}
              onClick={() =>
                dispatch(
                  setSeedWeights(seedWeightsToString(invocation.variations))
                )
              }
            />
          )}
          {invocation.seamless && (
            <MetadataItem
              label="Seamless"
              value={invocation.seamless}
              onClick={() => dispatch(setSeamless(invocation.seamless))}
            />
          )}
          {invocation.hires_fix && (
            <MetadataItem
              label="High Resolution Optimization"
              value={invocation.hires_fix}
              onClick={() => dispatch(setHiresFix(invocation.hires_fix))}
            />
          )}
          {invocation.width && (
            <MetadataItem
              label="Width"
              value={invocation.width}
              onClick={() => dispatch(setWidth(invocation.width))}
            />
          )}
          {invocation.height && (
            <MetadataItem
              label="Height"
              value={invocation.height}
              onClick={() => dispatch(setHeight(invocation.height))}
            />
          )}
          {/* {init_image_path && (
            <MetadataItem
              label="Initial image"
              value={init_image_path}
              isLink
              onClick={() => dispatch(setInitialImage(init_image_path))}
            />
          )} */}
          {invocation.strength && (
            <MetadataItem
              label="Image to image strength"
              value={invocation.strength}
              onClick={() => dispatch(setImg2imgStrength(invocation.strength))}
            />
          )}
          {invocation.fit && (
            <MetadataItem
              label="Image to image fit"
              value={invocation.fit}
              onClick={() =>
                dispatch(setShouldFitToWidthHeight(invocation.fit))
              }
            />
          )}
        </>
      ) : (
        <Center width="100%" pt={10}>
          <Text fontSize="lg" fontWeight="semibold">
            No metadata available
          </Text>
        </Center>
      )}
      <Flex gap={2} direction="column">
        <Flex gap={2}>
          <Tooltip label="Copy metadata JSON">
            <IconButton
              aria-label={t('accessibility.copyMetadataJson')}
              icon={<FaCopy />}
              size="xs"
              variant="ghost"
              fontSize={14}
              onClick={() => navigator.clipboard.writeText(metadataJSON)}
            />
          </Tooltip>
          <Text fontWeight="semibold">Metadata JSON:</Text>
        </Flex>
        <Box
          sx={{
            mt: 0,
            mr: 2,
            mb: 4,
            ml: 2,
            padding: 4,
            borderRadius: 'base',
            overflowX: 'scroll',
            wordBreak: 'break-all',
            bg: 'whiteAlpha.500',
            _dark: { bg: 'blackAlpha.500' },
          }}
        >
          <pre>{metadataJSON}</pre>
        </Box>
      </Flex>
    </Flex>
  );
}, memoEqualityCheck);

ImageMetadataViewer.displayName = 'ImageMetadataViewer';

export default ImageMetadataViewer;
