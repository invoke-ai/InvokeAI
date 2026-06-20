import type { GeneratedImageContract, WidgetRuntimeApi } from '@workbench/types';

import { Badge, Box, Flex, HStack, Stack } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { ArrowLeftRightIcon, Columns2Icon, XIcon } from 'lucide-react';
import { useEffect, useEffectEvent, useRef, useState, type CSSProperties } from 'react';

type CompareMode = 'slider' | 'side-by-side';

const COMPARE_IMAGE_STYLE: CSSProperties = {
  display: 'block',
  height: '100%',
  inset: 0,
  objectFit: 'contain',
  pointerEvents: 'none',
  position: 'absolute',
  userSelect: 'none',
  width: '100%',
};

const previewGridCss = {
  backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1.5px)',
  backgroundPosition: 'center',
  backgroundRepeat: 'repeat',
  backgroundSize: '24px 24px',
} as const;

/**
 * Two-image comparison: a draggable slider that reveals the compare image over
 * the selected one, or a side-by-side split.
 */
export const PreviewCompare = ({
  baseImage,
  compareImage,
  onExit,
  onSwap,
  runtime,
}: {
  baseImage: GeneratedImageContract;
  compareImage: GeneratedImageContract;
  onExit: () => void;
  onSwap: () => void;
  runtime: WidgetRuntimeApi;
}) => {
  const [mode, setMode] = useState<CompareMode>('slider');
  const [dividerPercent, setDividerPercent] = useState(50);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const isDraggingRef = useRef(false);
  const nextMode = useEffectEvent(() => setMode((current) => (current === 'slider' ? 'side-by-side' : 'slider')));

  useEffect(() => {
    const command = runtime.commands.register({
      handler: nextMode,
      id: 'viewer.nextComparisonMode',
      title: 'Next comparison mode',
    });
    const hotkey = runtime.hotkeys.register({
      commandId: 'viewer.nextComparisonMode',
      defaultKeys: ['m'],
      id: 'viewer.nextComparisonMode',
      title: 'Next comparison mode',
    });

    return () => {
      command();
      hotkey();
    };
  }, [runtime.commands, runtime.hotkeys]);

  const updateDivider = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();

    if (!rect || rect.width === 0) {
      return;
    }

    setDividerPercent(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  };

  return (
    <Stack gap="3" h="full" minH="0" w="full">
      <Flex
        align="center"
        backgroundColor="bg.inset"
        borderWidth="1px"
        borderColor="border.subtle"
        color="fg.grid"
        css={previewGridCss}
        flex="1"
        justify="center"
        minH="0"
        overflow="hidden"
        p="6"
        rounded="lg"
        w="full"
      >
        {mode === 'slider' ? (
          <Box
            ref={containerRef}
            bg="transparent"
            cursor="ew-resize"
            h="full"
            overflow="hidden"
            position="relative"
            style={{ touchAction: 'none' }}
            w="full"
            onPointerDown={(event) => {
              if (event.pointerType === 'mouse' && event.button !== 0) {
                return;
              }

              event.preventDefault();
              isDraggingRef.current = true;
              event.currentTarget.setPointerCapture(event.pointerId);
              updateDivider(event.clientX);
            }}
            onPointerMove={(event) => {
              if (isDraggingRef.current) {
                event.preventDefault();
                updateDivider(event.clientX);
              }
            }}
            onPointerUp={(event) => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }

              isDraggingRef.current = false;
            }}
            onPointerCancel={() => {
              isDraggingRef.current = false;
            }}
            onLostPointerCapture={() => {
              isDraggingRef.current = false;
            }}
          >
            <img alt={baseImage.imageName} draggable={false} src={baseImage.imageUrl} style={COMPARE_IMAGE_STYLE} />
            <Box
              bg="transparent"
              inset="0"
              pointerEvents="none"
              position="absolute"
              style={{ clipPath: `inset(0 ${100 - dividerPercent}% 0 0)` }}
            >
              <img
                alt={compareImage.imageName}
                draggable={false}
                src={compareImage.imageUrl}
                style={COMPARE_IMAGE_STYLE}
              />
            </Box>
            <Box
              bg="accent.solid"
              bottom="0"
              left={`${dividerPercent}%`}
              pointerEvents="none"
              position="absolute"
              top="0"
              transform="translateX(-50%)"
              w="2px"
            />
            <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
              Compare
            </Badge>
            <Badge pointerEvents="none" position="absolute" right="2" size="xs" top="2" variant="solid">
              Viewing
            </Badge>
          </Box>
        ) : (
          <HStack gap="2" h="full" minH="0" w="full">
            <CompareSidePane image={compareImage} label="Compare" />
            <CompareSidePane image={baseImage} label="Viewing" />
          </HStack>
        )}
      </Flex>
      <HStack flexShrink={0} gap="1" justify="center">
        <Button size="2xs" variant="outline" onClick={() => setMode(mode === 'slider' ? 'side-by-side' : 'slider')}>
          <Columns2Icon />
          {mode === 'slider' ? 'Side by Side' : 'Slider'}
        </Button>
        <Button size="2xs" variant="outline" onClick={onSwap}>
          <ArrowLeftRightIcon />
          Swap
        </Button>
        <Button size="2xs" variant="outline" onClick={onExit}>
          <XIcon />
          Exit Compare
        </Button>
      </HStack>
    </Stack>
  );
};

const CompareSidePane = ({ image, label }: { image: GeneratedImageContract; label: string }) => (
  <Box bg="transparent" flex="1" h="full" minW="0" overflow="hidden" position="relative">
    <img alt={image.imageName} draggable={false} src={image.imageUrl} style={COMPARE_IMAGE_STYLE} />
    <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
      {label}
    </Badge>
  </Box>
);
