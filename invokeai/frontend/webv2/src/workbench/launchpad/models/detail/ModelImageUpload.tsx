/* eslint-disable react/react-compiler, react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig } from '@workbench/models/types';

import { Box, Flex, Icon, Image, Stack, Text } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { deleteModelImage, getModelImageUrl, updateModelImage } from '@workbench/models/api';
import { markCoverImageChanged, useModelsSelector } from '@workbench/models/modelsStore';
import { ImageIcon, UploadIcon, XIcon } from 'lucide-react';
import { useEffect, useRef, useState, type DragEvent } from 'react';

const ACCEPTED_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);

const dragContainsFiles = (event: DragEvent): boolean => Array.from(event.dataTransfer.types).includes('Files');

/**
 * The model's cover image as one large dropzone: click anywhere on the tile
 * or drop an image file onto it to upload. The remove button sits in the top
 * corner and is hover/focus-revealed (and inert while hidden).
 */
export const ModelImageUpload = ({
  model,
  onError,
  onUpdated,
}: {
  model: Pick<ModelConfig, 'cover_image' | 'key' | 'name'>;
  onError: (message: string) => void;
  onUpdated: () => void;
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const dragDepthRef = useRef(0);
  const imageVersion = useModelsSelector((snapshot) => snapshot.coverImageVersions[model.key]);
  const [hasImage, setHasImage] = useState(Boolean(model.cover_image));
  const [isBusy, setIsBusy] = useState(false);
  const [isDropActive, setIsDropActive] = useState(false);

  useEffect(() => {
    dragDepthRef.current = 0;
    setHasImage(Boolean(model.cover_image));
    setIsBusy(false);
    setIsDropActive(false);
  }, [model.cover_image, model.key]);

  const handleFile = async (file: File | undefined) => {
    if (!file) {
      return;
    }

    if (!ACCEPTED_TYPES.has(file.type)) {
      onError('Use a PNG, JPEG, or WebP image.');
      return;
    }

    setIsBusy(true);
    const modelKey = model.key;

    try {
      await updateModelImage(modelKey, file);
      setHasImage(true);
      // Bumps the cache-bust version so this tile and list thumbnails reload.
      markCoverImageChanged(modelKey, true);
      onUpdated();
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Failed to upload model image.');
    } finally {
      setIsBusy(false);
    }
  };

  const handleDelete = async () => {
    setIsBusy(true);
    const modelKey = model.key;

    try {
      await deleteModelImage(modelKey);
      setHasImage(false);
      markCoverImageChanged(modelKey, false);
      onUpdated();
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Failed to remove model image.');
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <Box
      aria-busy={isBusy || undefined}
      aria-label={`Upload cover image for ${model.name}`}
      bg="bg.emphasized"
      borderColor={isDropActive ? 'accent.solid' : 'border.subtle'}
      borderStyle={hasImage ? 'solid' : 'dashed'}
      borderWidth={isDropActive ? '2px' : '1px'}
      boxSize="24"
      cursor="pointer"
      flexShrink={0}
      overflow="hidden"
      position="relative"
      role="button"
      rounded="lg"
      tabIndex={0}
      className="group"
      _focusVisible={{ outline: '2px solid {colors.accent.solid}', outlineOffset: '2px' }}
      onClick={() => {
        if (!isBusy) {
          inputRef.current?.click();
        }
      }}
      onKeyDown={(event) => {
        if ((event.key === 'Enter' || event.key === ' ') && !isBusy) {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragEnter={(event) => {
        if (!dragContainsFiles(event)) {
          return;
        }

        event.preventDefault();
        dragDepthRef.current += 1;
        setIsDropActive(true);
      }}
      onDragLeave={(event) => {
        if (!dragContainsFiles(event)) {
          return;
        }

        dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);

        if (dragDepthRef.current === 0) {
          setIsDropActive(false);
        }
      }}
      onDragOver={(event) => {
        if (dragContainsFiles(event)) {
          event.preventDefault();
        }
      }}
      onDrop={(event) => {
        if (!dragContainsFiles(event)) {
          return;
        }

        event.preventDefault();
        dragDepthRef.current = 0;
        setIsDropActive(false);
        void handleFile(event.dataTransfer.files[0]);
      }}
    >
      <input
        ref={inputRef}
        accept="image/png,image/jpeg,image/webp"
        hidden
        type="file"
        onChange={(event) => {
          void handleFile(event.currentTarget.files?.[0] ?? undefined);
          event.currentTarget.value = '';
        }}
        onClick={(event) => event.stopPropagation()}
      />
      {hasImage ? (
        <Image
          alt={`${model.name} cover`}
          boxSize="full"
          fit="cover"
          src={getModelImageUrl(model.key, imageVersion ? String(imageVersion) : undefined)}
          onError={() => setHasImage(false)}
        />
      ) : (
        <Stack align="center" boxSize="full" color="fg.subtle" gap="1" justify="center">
          <Icon as={ImageIcon} boxSize="5" />
          <Text fontSize="2xs">Add image</Text>
        </Stack>
      )}
      {isDropActive ? (
        <Flex
          align="center"
          bg="bg.muted/85"
          boxSize="full"
          inset="0"
          justify="center"
          pointerEvents="none"
          position="absolute"
        >
          <Icon as={UploadIcon} boxSize="5" color="accent.solid" />
        </Flex>
      ) : null}
      {hasImage && !isDropActive ? (
        <Box
          opacity={0}
          pointerEvents="none"
          position="absolute"
          right="1"
          top="1"
          transition="opacity var(--wb-motion-duration-fast) ease"
          _groupFocusWithin={{ opacity: 1, pointerEvents: 'auto' }}
          _groupHover={{ opacity: 1, pointerEvents: 'auto' }}
        >
          <Tooltip content="Remove image">
            <IconButton
              aria-label="Remove model image"
              colorPalette="red"
              size="2xs"
              variant="solid"
              onClick={(event) => {
                event.stopPropagation();
                void handleDelete();
              }}
            >
              <Icon as={XIcon} boxSize="3" />
            </IconButton>
          </Tooltip>
        </Box>
      ) : null}
    </Box>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
