import { ExternalLinkIcon } from '@chakra-ui/icons';
import {
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
import promptToString from 'common/util/promptToString';
import { seedWeightsToString } from 'common/util/seedWeightPairs';
import {
  setCfgScale,
  setHeight,
  setImg2imgStrength,
  setInitialImage,
  setMaskPath,
  setPerlin,
  setPrompt,
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
  return (
    <Flex gap={2}>
      {onClick && (
        <Tooltip label={`Recall ${label}`}>
          <IconButton
            aria-label="Use this parameter"
            icon={<IoArrowUndoCircleOutline />}
            size={'xs'}
            variant={'ghost'}
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
            size={'xs'}
            variant={'ghost'}
            fontSize={14}
            onClick={() => navigator.clipboard.writeText(value.toString())}
          />
        </Tooltip>
      )}
      <Flex direction={labelPosition ? 'column' : 'row'}>
        <Text fontWeight={'semibold'} whiteSpace={'pre-wrap'} pr={2}>
          {label}:
        </Text>
        {isLink ? (
          <Link href={value.toString()} isExternal wordBreak={'break-all'}>
            {value.toString()} <ExternalLinkIcon mx="2px" />
          </Link>
        ) : (
          <Text overflowY={'scroll'} wordBreak={'break-all'}>
            {value.toString()}
          </Text>
        )}
      </Flex>
    </Flex>
  );
};

type ImageMetadataViewerProps = {
  image: InvokeAI.Image;
  styleClass?: string;
};

// TODO: I don't know if this is needed.
const memoEqualityCheck = (
  prev: ImageMetadataViewerProps,
  next: ImageMetadataViewerProps
) => prev.image.uuid === next.image.uuid;

// TODO: Show more interesting information in this component.

/**
 * Image metadata viewer overlays currently selected image and provides
 * access to any of its metadata for use in processing.
 */
const ImageMetadataViewer = memo(
  ({ image, styleClass }: ImageMetadataViewerProps) => {
    const dispatch = useAppDispatch();

    useHotkeys('esc', () => {
      dispatch(setShouldShowImageDetails(false));
    });

    const metadata = image?.metadata?.image || {};
    const dreamPrompt = image?.dreamPrompt;

    const {
      cfg_scale,
      fit,
      height,
      hires_fix,
      init_image_path,
      mask_image_path,
      orig_path,
      perlin,
      postprocessing,
      prompt,
      sampler,
      seamless,
      seed,
      steps,
      strength,
      denoise_str,
      threshold,
      type,
      variations,
      width,
    } = metadata;

    const metadataJSON = JSON.stringify(image.metadata, null, 2);

    return (
      <div className={`image-metadata-viewer ${styleClass}`}>
        <Flex gap={1} direction={'column'} width={'100%'}>
          <Flex gap={2}>
            <Text fontWeight={'semibold'}>File:</Text>
            <Link href={image.url} isExternal maxW={'calc(100% - 3rem)'}>
              {image.url.length > 64
                ? image.url.substring(0, 64).concat('...')
                : image.url}
              <ExternalLinkIcon mx="2px" />
            </Link>
          </Flex>
          {Object.keys(metadata).length > 0 ? (
            <>
              {type && <MetadataItem label="Generation type" value={type} />}
              {image.metadata?.model_weights && (
                <MetadataItem
                  label="Model"
                  value={image.metadata.model_weights}
                />
              )}
              {['esrgan', 'gfpgan'].includes(type) && (
                <MetadataItem label="Original image" value={orig_path} />
              )}
              {prompt && (
                <MetadataItem
                  label="Prompt"
                  labelPosition="top"
                  value={promptToString(prompt)}
                  onClick={() => dispatch(setPrompt(prompt))}
                />
              )}
              {seed !== undefined && (
                <MetadataItem
                  label="Seed"
                  value={seed}
                  onClick={() => dispatch(setSeed(seed))}
                />
              )}
              {threshold !== undefined && (
                <MetadataItem
                  label="Noise Threshold"
                  value={threshold}
                  onClick={() => dispatch(setThreshold(threshold))}
                />
              )}
              {perlin !== undefined && (
                <MetadataItem
                  label="Perlin Noise"
                  value={perlin}
                  onClick={() => dispatch(setPerlin(perlin))}
                />
              )}
              {sampler && (
                <MetadataItem
                  label="Sampler"
                  value={sampler}
                  onClick={() => dispatch(setSampler(sampler))}
                />
              )}
              {steps && (
                <MetadataItem
                  label="Steps"
                  value={steps}
                  onClick={() => dispatch(setSteps(steps))}
                />
              )}
              {cfg_scale !== undefined && (
                <MetadataItem
                  label="CFG scale"
                  value={cfg_scale}
                  onClick={() => dispatch(setCfgScale(cfg_scale))}
                />
              )}
              {variations && variations.length > 0 && (
                <MetadataItem
                  label="Seed-weight pairs"
                  value={seedWeightsToString(variations)}
                  onClick={() =>
                    dispatch(setSeedWeights(seedWeightsToString(variations)))
                  }
                />
              )}
              {seamless && (
                <MetadataItem
                  label="Seamless"
                  value={seamless}
                  onClick={() => dispatch(setSeamless(seamless))}
                />
              )}
              {hires_fix && (
                <MetadataItem
                  label="High Resolution Optimization"
                  value={hires_fix}
                  onClick={() => dispatch(setHiresFix(hires_fix))}
                />
              )}
              {width && (
                <MetadataItem
                  label="Width"
                  value={width}
                  onClick={() => dispatch(setWidth(width))}
                />
              )}
              {height && (
                <MetadataItem
                  label="Height"
                  value={height}
                  onClick={() => dispatch(setHeight(height))}
                />
              )}
              {init_image_path && (
                <MetadataItem
                  label="Initial image"
                  value={init_image_path}
                  isLink
                  onClick={() => dispatch(setInitialImage(init_image_path))}
                />
              )}
              {mask_image_path && (
                <MetadataItem
                  label="Mask image"
                  value={mask_image_path}
                  isLink
                  onClick={() => dispatch(setMaskPath(mask_image_path))}
                />
              )}
              {type === 'img2img' && strength && (
                <MetadataItem
                  label="Image to image strength"
                  value={strength}
                  onClick={() => dispatch(setImg2imgStrength(strength))}
                />
              )}
              {fit && (
                <MetadataItem
                  label="Image to image fit"
                  value={fit}
                  onClick={() => dispatch(setShouldFitToWidthHeight(fit))}
                />
              )}
              {postprocessing && postprocessing.length > 0 && (
                <>
                  <Heading size={'sm'}>Postprocessing</Heading>
                  {postprocessing.map(
                    (
                      postprocess: InvokeAI.PostProcessedImageMetadata,
                      i: number
                    ) => {
                      if (postprocess.type === 'esrgan') {
                        const { scale, strength, denoise_str } = postprocess;
                        return (
                          <Flex
                            key={i}
                            pl={'2rem'}
                            gap={1}
                            direction={'column'}
                          >
                            <Text size={'md'}>{`${
                              i + 1
                            }: Upscale (ESRGAN)`}</Text>
                            <MetadataItem
                              label="Scale"
                              value={scale}
                              onClick={() => dispatch(setUpscalingLevel(scale))}
                            />
                            <MetadataItem
                              label="Strength"
                              value={strength}
                              onClick={() =>
                                dispatch(setUpscalingStrength(strength))
                              }
                            />
                            {denoise_str !== undefined && (
                              <MetadataItem
                                label="Denoising strength"
                                value={denoise_str}
                                onClick={() =>
                                  dispatch(setUpscalingDenoising(denoise_str))
                                }
                              />
                            )}
                          </Flex>
                        );
                      } else if (postprocess.type === 'gfpgan') {
                        const { strength } = postprocess;
                        return (
                          <Flex
                            key={i}
                            pl={'2rem'}
                            gap={1}
                            direction={'column'}
                          >
                            <Text size={'md'}>{`${
                              i + 1
                            }: Face restoration (GFPGAN)`}</Text>

                            <MetadataItem
                              label="Strength"
                              value={strength}
                              onClick={() => {
                                dispatch(setFacetoolStrength(strength));
                                dispatch(setFacetoolType('gfpgan'));
                              }}
                            />
                          </Flex>
                        );
                      } else if (postprocess.type === 'codeformer') {
                        const { strength, fidelity } = postprocess;
                        return (
                          <Flex
                            key={i}
                            pl={'2rem'}
                            gap={1}
                            direction={'column'}
                          >
                            <Text size={'md'}>{`${
                              i + 1
                            }: Face restoration (Codeformer)`}</Text>

                            <MetadataItem
                              label="Strength"
                              value={strength}
                              onClick={() => {
                                dispatch(setFacetoolStrength(strength));
                                dispatch(setFacetoolType('codeformer'));
                              }}
                            />
                            {fidelity && (
                              <MetadataItem
                                label="Fidelity"
                                value={fidelity}
                                onClick={() => {
                                  dispatch(setCodeformerFidelity(fidelity));
                                  dispatch(setFacetoolType('codeformer'));
                                }}
                              />
                            )}
                          </Flex>
                        );
                      }
                    }
                  )}
                </>
              )}
              {dreamPrompt && (
                <MetadataItem
                  withCopy
                  label="Dream Prompt"
                  value={dreamPrompt}
                />
              )}
              <Flex gap={2} direction={'column'}>
                <Flex gap={2}>
                  <Tooltip label={`Copy metadata JSON`}>
                    <IconButton
                      aria-label="Copy metadata JSON"
                      icon={<FaCopy />}
                      size={'xs'}
                      variant={'ghost'}
                      fontSize={14}
                      onClick={() =>
                        navigator.clipboard.writeText(metadataJSON)
                      }
                    />
                  </Tooltip>
                  <Text fontWeight={'semibold'}>Metadata JSON:</Text>
                </Flex>
                <div className={'image-json-viewer'}>
                  <pre>{metadataJSON}</pre>
                </div>
              </Flex>
            </>
          ) : (
            <Center width={'100%'} pt={10}>
              <Text fontSize={'lg'} fontWeight="semibold">
                No metadata available
              </Text>
            </Center>
          )}
        </Flex>
      </div>
    );
  },
  memoEqualityCheck
);

ImageMetadataViewer.displayName = 'ImageMetadataViewer';

export default ImageMetadataViewer;
