import { HStack, Icon, Text } from '@chakra-ui/react';
import { Tooltip } from '@platform/ui/Tooltip';
import { LockIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** What keeps the canvas from editing: generation still running, or results waiting to be accepted or discarded. */
export type CanvasLockReason = 'generating' | 'reviewing';

const CHIP_TOP = 'var(--wb-center-chrome-inset, var(--chakra-spacing-2))';

const LOCK_KEYS: Record<CanvasLockReason, { help: string; label: string; tool: string }> = {
  generating: {
    help: 'widgets.canvas.lock.generatingHelp',
    label: 'widgets.canvas.lock.generating',
    tool: 'widgets.canvas.lock.toolGenerating',
  },
  reviewing: {
    help: 'widgets.canvas.lock.reviewingHelp',
    label: 'widgets.canvas.lock.reviewing',
    tool: 'widgets.canvas.lock.toolReviewing',
  },
};

/** The reason a locked tool button gives in its tooltip. */
export const canvasLockToolReasonKey = (reason: CanvasLockReason): string => LOCK_KEYS[reason].tool;

/**
 * Says the canvas is locked, and why, while a generation runs or its results wait for review. Only View works
 * meanwhile; the help (a tooltip, and part of the status) says how the lock ends. A status, so it is announced.
 */
export const CanvasLockChip = ({ reason }: { reason: CanvasLockReason }) => {
  const { t } = useTranslation();
  const keys = LOCK_KEYS[reason];
  return (
    <Tooltip content={t(keys.help)}>
      <HStack
        aria-live="polite"
        bg="bg.subtle"
        borderColor="border.subtle"
        borderWidth="1px"
        color="fg.muted"
        focusVisibleRing="outside"
        gap="1.5"
        left="50%"
        position="absolute"
        px="2.5"
        py="1"
        role="status"
        rounded="full"
        shadow="sm"
        // Focusable so the help tooltip opens from the keyboard too.
        tabIndex={0}
        top={CHIP_TOP}
        transform="translateX(-50%)"
        zIndex="2"
      >
        <Icon as={LockIcon} aria-hidden boxSize="3" />
        <Text fontSize="sm" fontWeight="600" whiteSpace="nowrap">
          {t(keys.label)}
        </Text>
        {/* The tooltip shows on hover or focus; the status reads its help too. */}
        <Text srOnly>{t(keys.help)}</Text>
      </HStack>
    </Tooltip>
  );
};
