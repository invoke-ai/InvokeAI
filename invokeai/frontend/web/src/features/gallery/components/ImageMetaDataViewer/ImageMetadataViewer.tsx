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
import * as InvokeAI from 'app/types/invokeai';
import { useAppDispatch } from 'app/store/storeHooks';
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
  const node = image.metadata.invokeai?.node as Record<string, any>;

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
        overflow: 'scroll',
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
      {node && Object.keys(node).length > 0 ? (
        <>
          {node.type && (
            <MetadataItem label="Invocation type" value={node.type} />
          )}
          {node.model && <MetadataItem label="Model" value={node.model} />}
          {node.prompt && (
            <MetadataItem
              label="Prompt"
              labelPosition="top"
              value={
                typeof node.prompt === 'string'
                  ? node.prompt
                  : promptToString(node.prompt)
              }
              onClick={() => setBothPrompts(node.prompt)}
            />
          )}
          {node.seed !== undefined && (
            <MetadataItem
              label="Seed"
              value={node.seed}
              onClick={() => dispatch(setSeed(Number(node.seed)))}
            />
          )}
          {node.threshold !== undefined && (
            <MetadataItem
              label="Noise Threshold"
              value={node.threshold}
              onClick={() => dispatch(setThreshold(Number(node.threshold)))}
            />
          )}
          {node.perlin !== undefined && (
            <MetadataItem
              label="Perlin Noise"
              value={node.perlin}
              onClick={() => dispatch(setPerlin(Number(node.perlin)))}
            />
          )}
          {node.scheduler && (
            <MetadataItem
              label="Sampler"
              value={node.scheduler}
              onClick={() => dispatch(setSampler(node.scheduler))}
            />
          )}
          {node.steps && (
            <MetadataItem
              label="Steps"
              value={node.steps}
              onClick={() => dispatch(setSteps(Number(node.steps)))}
            />
          )}
          {node.cfg_scale !== undefined && (
            <MetadataItem
              label="CFG scale"
              value={node.cfg_scale}
              onClick={() => dispatch(setCfgScale(Number(node.cfg_scale)))}
            />
          )}
          {node.variations && node.variations.length > 0 && (
            <MetadataItem
              label="Seed-weight pairs"
              value={seedWeightsToString(node.variations)}
              onClick={() =>
                dispatch(setSeedWeights(seedWeightsToString(node.variations)))
              }
            />
          )}
          {node.seamless && (
            <MetadataItem
              label="Seamless"
              value={node.seamless}
              onClick={() => dispatch(setSeamless(node.seamless))}
            />
          )}
          {node.hires_fix && (
            <MetadataItem
              label="High Resolution Optimization"
              value={node.hires_fix}
              onClick={() => dispatch(setHiresFix(node.hires_fix))}
            />
          )}
          {node.width && (
            <MetadataItem
              label="Width"
              value={node.width}
              onClick={() => dispatch(setWidth(Number(node.width)))}
            />
          )}
          {node.height && (
            <MetadataItem
              label="Height"
              value={node.height}
              onClick={() => dispatch(setHeight(Number(node.height)))}
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
          {node.strength && (
            <MetadataItem
              label="Image to image strength"
              value={node.strength}
              onClick={() =>
                dispatch(setImg2imgStrength(Number(node.strength)))
              }
            />
          )}
          {node.fit && (
            <MetadataItem
              label="Image to image fit"
              value={node.fit}
              onClick={() => dispatch(setShouldFitToWidthHeight(node.fit))}
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
