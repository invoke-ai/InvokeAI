/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import { Checkbox, HStack, VisuallyHidden } from '@chakra-ui/react';
import { Tooltip } from '@platform/ui';
import { useId } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Choices applied to the next install from a source. FP8 storage is an opt-in only: identification already turns it
 * on for checkpoints whose weights are stored in FP8, so leaving the box unticked never turns that off.
 */
export const InstallOptions = ({
  fp8Storage,
  inplace,
  onSetFp8Storage,
  onSetInplace,
}: {
  fp8Storage: boolean;
  /** Omitted, with its setter, for sources that cannot be installed in place (URLs, repos). */
  inplace?: boolean;
  onSetFp8Storage: (fp8Storage: boolean) => void;
  onSetInplace?: (inplace: boolean) => void;
}) => {
  const { t } = useTranslation();
  // The tooltip describes its trigger, the label; the focusable input needs the same text to be announced.
  const fp8StorageDescriptionId = useId();

  return (
    <HStack gap="3">
      {inplace === undefined || onSetInplace === undefined ? null : (
        <Checkbox.Root
          checked={inplace}
          colorPalette="accent"
          size="xs"
          onCheckedChange={(event) => onSetInplace(event.checked === true)}
        >
          <Checkbox.HiddenInput />
          <Checkbox.Control />
          <Checkbox.Label fontSize="2xs">{t('models.installInPlace')}</Checkbox.Label>
        </Checkbox.Root>
      )}
      <Tooltip content={t('models.installFp8StorageTooltip')}>
        <Checkbox.Root
          checked={fp8Storage}
          colorPalette="accent"
          size="xs"
          onCheckedChange={(event) => onSetFp8Storage(event.checked === true)}
        >
          <Checkbox.HiddenInput aria-describedby={fp8StorageDescriptionId} />
          <Checkbox.Control />
          <Checkbox.Label fontSize="2xs">{t('models.installFp8Storage')}</Checkbox.Label>
        </Checkbox.Root>
      </Tooltip>
      <VisuallyHidden id={fp8StorageDescriptionId}>{t('models.installFp8StorageTooltip')}</VisuallyHidden>
    </HStack>
  );
};
