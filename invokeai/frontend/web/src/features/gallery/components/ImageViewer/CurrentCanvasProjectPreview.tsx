import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { CanvasProjectDTO } from 'services/api/types';

type Props = {
  projectDTO: CanvasProjectDTO | null;
};

/**
 * Viewer preview for a saved canvas project. Shows the preview WebP at full size, with a hint
 * to use the "Load Project" toolbar button to actually restore the canvas state.
 *
 * Unlike CurrentImagePreview, this is read-only: there's no DnD target, no compare, no progress
 * overlay — a project is a discrete restore action triggered from the toolbar.
 */
export const CurrentCanvasProjectPreview = memo(({ projectDTO }: Props) => {
  const { t } = useTranslation();

  if (!projectDTO) {
    return null;
  }

  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center" position="relative" flexDir="column" gap={2}>
      {projectDTO.thumbnail_url ? (
        <Image
          src={projectDTO.thumbnail_url}
          objectFit="contain"
          maxW="full"
          maxH="full"
          borderRadius="base"
          pointerEvents="none"
        />
      ) : (
        <Flex
          w="80%"
          h="80%"
          alignItems="center"
          justifyContent="center"
          bg="base.800"
          borderRadius="base"
          color="base.400"
        >
          <Text>{t('controlLayers.canvasProject.project')}</Text>
        </Flex>
      )}
    </Flex>
  );
});

CurrentCanvasProjectPreview.displayName = 'CurrentCanvasProjectPreview';
