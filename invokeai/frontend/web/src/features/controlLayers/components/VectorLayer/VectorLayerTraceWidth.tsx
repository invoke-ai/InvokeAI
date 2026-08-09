import { FormControl, FormLabel } from '@invoke-ai/ui-library';
import { ToolWidthPicker } from 'features/controlLayers/components/Tool/ToolWidthPicker';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const VectorLayerTraceWidth = memo(() => {
  const { t } = useTranslation();
  const label = t('controlLayers.vectorEdit.traceWidth');

  return (
    <FormControl flex={1} minW={0} gap={2}>
      <FormLabel m={0} mt={1} whiteSpace="nowrap">
        {label}
      </FormLabel>
      <ToolWidthPicker mode="trace" ariaLabel={label} />
    </FormControl>
  );
});

VectorLayerTraceWidth.displayName = 'VectorLayerTraceWidth';
