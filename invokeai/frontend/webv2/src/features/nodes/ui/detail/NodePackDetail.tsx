/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { NodePackInfo } from '@features/nodes/core/catalog';

import { Badge, Box, Flex, HStack, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { isProblemPack } from '@features/nodes/core/library';
import { ensureInvocationTemplatesLoaded, useInvocationTemplatesSelector } from '@features/workflow/react';
import { Button } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { BlocksIcon, TriangleAlertIcon, Trash2Icon } from 'lucide-react';
import { useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { NodePreviewCard } from './NodePreviewCard';

/**
 * Build previews from backend invocation templates; nodeTypes are invocation keys and nodePack identifies their
 * owning pack. The host owns the uninstall confirmation because an uninstall unmounts this detail before the dialog
 * finishes closing.
 */
export const NodePackDetail = ({
  onRequestUninstall,
  pack,
}: {
  onRequestUninstall: (pack: NodePackInfo) => void;
  pack: NodePackInfo;
}) => {
  const { t } = useTranslation();
  const status = useInvocationTemplatesSelector((snapshot) => snapshot.status);
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);

  useEffect(() => {
    ensureInvocationTemplatesLoaded();
  }, []);

  const packTemplates = useMemo(
    () => pack.nodeTypes.map((nodeType) => templates[nodeType]).filter((template) => template !== undefined),
    [pack.nodeTypes, templates]
  );

  const isLoadingTemplates = (status === 'idle' || status === 'loading') && packTemplates.length === 0;

  return (
    <Stack gap="4" pb="4">
      <HStack align="start" gap="3" justify="space-between">
        <Stack flex="1" gap="1.5" minW="0">
          <HStack gap="2" minW="0">
            <Icon as={BlocksIcon} boxSize="4" color="fg.muted" flexShrink={0} />
            <MiddleTruncate fontSize="lg" fontWeight="700" minW="0" text={pack.name} />
          </HStack>
          <Text color="fg.muted" fontFamily="mono" fontSize="xs" overflowWrap="anywhere">
            {pack.path}
          </Text>
          <HStack gap="1.5" wrap="wrap">
            {isProblemPack(pack) ? (
              <Badge colorPalette="orange" fontSize="xs" variant="surface">
                {t('nodes.noNodesRegistered')}
              </Badge>
            ) : (
              <Badge colorPalette="blue" fontSize="xs" variant="surface">
                {t('nodes.nodeCount', { count: pack.nodeCount })}
              </Badge>
            )}
          </HStack>
          {isProblemPack(pack) ? (
            <Text color="fg.muted" fontSize="xs">
              {t('nodes.noNodesRegisteredHint')}
            </Text>
          ) : null}
        </Stack>
        <Button colorPalette="red" flexShrink={0} variant="outline" onClick={() => onRequestUninstall(pack)}>
          <Icon as={Trash2Icon} boxSize="3" />
          {t('nodes.uninstall')}
        </Button>
      </HStack>

      <Stack gap="2">
        <Text color="fg.muted" fontSize="xs" fontWeight="600" textTransform="uppercase">
          {t('nodes.nodesInPack')}
        </Text>
        {isLoadingTemplates ? (
          <Flex align="center" justify="center" py="10">
            <Spinner color="fg.subtle" size="lg" />
          </Flex>
        ) : packTemplates.length === 0 ? (
          <EmptyState
            description={t('nodes.noPreviewsDescription')}
            icon={<Icon as={TriangleAlertIcon} />}
            title={t('nodes.noPreviews')}
          />
        ) : (
          <Flex gap="8" wrap="wrap" px="2">
            {packTemplates.map((template) => (
              <Box key={template.type} flexShrink={0} maxW="full" w="18rem">
                <NodePreviewCard template={template} />
              </Box>
            ))}
          </Flex>
        )}
      </Stack>
    </Stack>
  );
};
